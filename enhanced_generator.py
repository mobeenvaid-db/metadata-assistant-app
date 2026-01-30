"""
Enhanced Metadata Generator
===========================

Advanced metadata generation with intelligent sampling, chunking, and context optimization.
Provides enterprise-grade generation capabilities inspired by dbxmetagen.

Version: 2.0 (Batch Processing)
"""

import json
import logging
import re
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import statistics
from collections import Counter
from pii_detector import PIIDetector

logger = logging.getLogger(__name__)

# Semaphore for limiting concurrent LLM calls (5 parallel requests max)
LLM_CONCURRENCY_LIMIT = 5

class EnhancedMetadataGenerator:
    """
    Enterprise-grade metadata generator with advanced sampling and analysis capabilities.
    Combines the power of LLM generation with intelligent data analysis.
    """
    
    def __init__(self, llm_service, unity_service, settings_manager=None):
        self.llm_service = llm_service
        self.unity_service = unity_service
        self.settings_manager = settings_manager
        self.pii_detector = PIIDetector(settings_manager, llm_service, unity_service)
        self.progress_callback = None
        self.progress_offset = 0  # For multi-model cumulative progress tracking
        
        # Performance Caches (shared across models to avoid redundant queries)
        self.pii_cache = {}  # Key: "catalog.schema.table.column" -> PII result
        self.column_metadata_cache = {}  # Key: "catalog.schema.table" -> [column_names]
        self.sample_data_cache = {}  # Key: "catalog.schema.table" -> sample data
        self.schema_tables_cache = {}  # Key: "catalog.schema" -> [table_info]
        
        # Resume on Failure Support
        self.incremental_save_callback = None  # Callback to save results incrementally
        self.completed_objects = set()  # Track completed objects for resume capability
        
        # Configuration - mirrors dbxmetagen variables.yml
        # Read sampling config from settings if available
        sampling_config = {}
        if settings_manager:
            try:
                sampling_config = settings_manager.get_sampling_config()
            except Exception as e:
                logger.warning(f"Could not load sampling config from settings: {e}")
                # If settings fail to load, disable sampling for safety
                sampling_config = {'enable_sampling': False, 'sample_rows': 10}
        else:
            # No settings manager available - disable sampling
            logger.warning("No settings manager available, disabling sampling")
            sampling_config = {'enable_sampling': False, 'sample_rows': 10}
        
        self.config = {
            'sample_rows': sampling_config.get('sample_rows', 10),
            'enable_sampling': sampling_config.get('enable_sampling', False),  # Default to disabled if not specified
            'redact_pii_in_samples': sampling_config.get('redact_pii_in_samples', False),
            'max_chunk_size': 10,
            'confidence_threshold': 0.7,
            'max_tokens': 512,
            'temperature': 0.3,
            'enable_pii_detection': True,
            'enable_data_profiling': True,
            'context_window': 3,  # Number of related objects to include for context
            # Intelligent batch sizing - read from settings
            'max_prompt_tokens': sampling_config.get('max_prompt_tokens', 4000),
            'max_batch_schemas': sampling_config.get('max_batch_schemas', 15),
            'max_batch_tables': sampling_config.get('max_batch_tables', 10),
            'max_batch_columns': sampling_config.get('max_batch_columns', 20),
            'estimated_tokens_per_schema': sampling_config.get('estimated_tokens_per_schema', 150),
            'estimated_tokens_per_table': sampling_config.get('estimated_tokens_per_table', 300),
            'estimated_tokens_per_column': sampling_config.get('estimated_tokens_per_column', 100),
            # NEW: Max columns to analyze for table context (prevents wide table slowdown)
            'max_table_context_columns': sampling_config.get('max_table_context_columns', 50)
        }
    
    def update_config(self, **kwargs):
        """Update generator configuration"""
        self.config.update(kwargs)
        logger.info(f"Updated config: {kwargs}")
    
    def _is_cancelled(self, run_id: str) -> bool:
        """Check if generation run has been cancelled by user"""
        try:
            # Check Flask app's cancellation flags
            from app_react import flask_app
            if hasattr(flask_app, 'generation_cancellations'):
                is_cancelled = run_id in flask_app.generation_cancellations
                if is_cancelled:
                    logger.info(f"🛑 Cancellation detected for run: {run_id}")
                return is_cancelled
        except Exception as e:
            logger.debug(f"Could not check cancellation status: {e}")
        return False
    
    def _estimate_prompt_tokens(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters for English)"""
        return len(text) // 4
    
    def _calculate_optimal_batch_size(self, items: List[Dict], item_type: str, max_batch_size: int) -> int:
        """
        Calculate optimal batch size based on token limits and item complexity.
        
        Args:
            items: List of items to batch
            item_type: 'schema', 'table', or 'column'
            max_batch_size: Maximum allowed batch size
        
        Returns:
            Optimal batch size that won't exceed token limits
        """
        if not items:
            return max_batch_size
        
        # Get token estimates from config
        tokens_per_item = self.config.get(f'estimated_tokens_per_{item_type}', 200)
        max_prompt_tokens = self.config.get('max_prompt_tokens', 4000)
        
        # Reserve tokens for prompt template (instructions, formatting, etc.)
        template_overhead = 500
        available_tokens = max_prompt_tokens - template_overhead
        
        # Calculate max items that fit in token budget
        token_based_limit = available_tokens // tokens_per_item
        
        # Use the smaller of token-based limit and configured max
        optimal_size = min(token_based_limit, max_batch_size)
        
        # For items with high complexity (many columns, samples), reduce further
        if item_type == 'table' and items:
            # Check if first table has many columns
            sample_item = items[0]
            column_count = sample_item.get('column_count', len(sample_item.get('columns', [])))
            if column_count > 20:
                # Reduce batch size for complex tables
                optimal_size = max(3, optimal_size // 2)
                logger.info(f"📊 Reducing table batch size to {optimal_size} due to high column count ({column_count})")
        
        return max(1, optimal_size)  # Ensure at least 1
    
    def _split_into_smart_batches(self, items: List[Any], item_type: str) -> List[List[Any]]:
        """
        Split items into intelligent batches that respect token limits.
        
        Args:
            items: List of items to batch
            item_type: 'schema', 'table', or 'column'
        
        Returns:
            List of batches, where each batch is a list of items
        """
        if not items:
            return []
        
        max_batch_size = self.config.get(f'max_batch_{item_type}s', 10)
        optimal_batch_size = self._calculate_optimal_batch_size(items, item_type, max_batch_size)
        
        batches = []
        for i in range(0, len(items), optimal_batch_size):
            batch = items[i:i+optimal_batch_size]
            batches.append(batch)
        
        logger.info(f"📦 Split {len(items)} {item_type}s into {len(batches)} batches (size: {optimal_batch_size} per batch)")
        return batches
    
    def _validate_prompt_size(self, prompt: str, batch_size: int, item_type: str) -> bool:
        """
        Validate that prompt size is within acceptable limits.
        Logs warning if prompt exceeds limits.
        
        Args:
            prompt: The prompt string to validate
            batch_size: Current batch size
            item_type: Type of items being processed
        
        Returns:
            True if prompt is acceptable, False if too large
        """
        estimated_tokens = self._estimate_prompt_tokens(prompt)
        max_tokens = self.config.get('max_prompt_tokens', 4000)
        
        if estimated_tokens > max_tokens:
            logger.warning(f"⚠️ Prompt size ({estimated_tokens} tokens) exceeds limit ({max_tokens} tokens) for {item_type} batch of size {batch_size}")
            logger.warning(f"⚠️ Prompt preview: {prompt[:200]}...")
            return False
        
        logger.info(f"✅ Prompt validated: {estimated_tokens} tokens for {batch_size} {item_type}s (limit: {max_tokens})")
        return True
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
        self._last_progress_update = 0  # Timestamp for throttling
        self._progress_throttle_seconds = 0.5  # Update max once per 0.5 seconds
    
    def set_incremental_save_callback(self, callback):
        """Set callback for incremental result saving (resume on failure support)"""
        self.incremental_save_callback = callback
    
    async def _save_checkpoint(self, results_batch: List[Dict], run_id: str):
        """
        Save a checkpoint of completed results (RESUME ON FAILURE support).
        
        This ensures users never lose work - even if the job times out after 450/700 objects,
        those 450 results are saved and can be resumed from.
        """
        if not self.incremental_save_callback or not results_batch:
            return
        
        try:
            # Save batch incrementally
            await self.incremental_save_callback(results_batch, run_id)
            
            # Track completed objects for resume
            for result in results_batch:
                self.completed_objects.add(result['full_name'])
                
            logger.info(f"💾 Checkpoint: Saved {len(results_batch)} results (total saved: {len(self.completed_objects)})")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Don't fail the entire generation if checkpoint fails - continue processing
    
    def _update_progress(self, **kwargs):
        """
        Update progress if callback is set (ZERO OVERHEAD with aggressive throttling).
        
        Throttles updates to max 2/second to avoid performance impact.
        Critical updates (phase changes) bypass throttle.
        """
        if not self.progress_callback:
            return
            
        try:
            import time
            current_time = time.time()
            
            # Allow critical updates (phase changes) to bypass throttle
            is_critical = 'current_phase' in kwargs
            
            # Throttle non-critical updates
            if not is_critical:
                time_since_last = current_time - self._last_progress_update
                if time_since_last < self._progress_throttle_seconds:
                    return  # Skip this update (too soon)
            
            # Apply progress offset for multi-model mode (cumulative progress)
            if 'processed_objects' in kwargs and self.progress_offset > 0:
                kwargs['processed_objects'] = kwargs['processed_objects'] + self.progress_offset
            
            # Only update if we have meaningful changes
            if kwargs:
                self.progress_callback(**kwargs)
                self._last_progress_update = current_time
                
        except Exception as e:
            # Use debug level to avoid log spam during generation
            logger.debug(f"Progress callback error: {e}")
    
    async def generate_enhanced_metadata(self, catalog_name: str, model: str, selected_objects: Dict = None, run_id: str = None, pii_model: str = None, progress_offset: int = 0) -> Dict:
        """
        Generate enhanced metadata for entire catalog with PII detection and data profiling
        
        Args:
            catalog_name: Name of the catalog
            model: LLM model for metadata generation
            selected_objects: Dict of selected schemas/tables/columns
            run_id: Unique run identifier
            pii_model: Optional LLM model for PII detection (defaults to metadata model)
        
        Note: Generation style is now controlled via Prompts settings in the Settings page
        """
        # Default pii_model to metadata model if not specified
        if not pii_model:
            pii_model = model
        
        # Store pii_model for use in nested methods
        self._current_pii_model = pii_model
        
        # Store progress offset for multi-model cumulative tracking
        self.progress_offset = progress_offset
            
        logger.info(f"Starting enhanced metadata generation for {catalog_name} (metadata: {model}, PII: {pii_model})")
        
        # Reload settings before each generation run to pick up any changes
        if self.settings_manager:
            try:
                sampling_config = self.settings_manager.get_sampling_config()
                # Update only the sampling-related settings
                self.config['sample_rows'] = sampling_config.get('sample_rows', self.config['sample_rows'])
                self.config['enable_sampling'] = sampling_config.get('enable_sampling', self.config['enable_sampling'])
                self.config['redact_pii_in_samples'] = sampling_config.get('redact_pii_in_samples', self.config['redact_pii_in_samples'])
                self.config['max_prompt_tokens'] = sampling_config.get('max_prompt_tokens', self.config['max_prompt_tokens'])
                self.config['max_batch_schemas'] = sampling_config.get('max_batch_schemas', self.config['max_batch_schemas'])
                self.config['max_batch_tables'] = sampling_config.get('max_batch_tables', self.config['max_batch_tables'])
                self.config['max_batch_columns'] = sampling_config.get('max_batch_columns', self.config['max_batch_columns'])
                self.config['estimated_tokens_per_schema'] = sampling_config.get('estimated_tokens_per_schema', self.config['estimated_tokens_per_schema'])
                self.config['estimated_tokens_per_table'] = sampling_config.get('estimated_tokens_per_table', self.config['estimated_tokens_per_table'])
                self.config['estimated_tokens_per_column'] = sampling_config.get('estimated_tokens_per_column', self.config['estimated_tokens_per_column'])
                self.config['max_table_context_columns'] = sampling_config.get('max_table_context_columns', self.config['max_table_context_columns'])
                logger.info(f"🔄 Reloaded settings: sample_rows={self.config['sample_rows']}, enable_sampling={self.config['enable_sampling']}")
            except Exception as e:
                logger.warning(f"Could not reload sampling config: {e}, using cached config")
        
        start_time = datetime.now()
        if run_id is None:
            run_id = f"enhanced_{start_time.strftime('%Y%m%d_%H%M%S')}_{catalog_name}"
        
        results = {
            'run_id': run_id,
            'catalog_name': catalog_name,
            'model': model,
            'started_at': start_time.isoformat(),
            'config': self.config.copy(),
            'summary': {
                'total_objects': 0,
                'processed_objects': 0,
                'pii_objects_detected': 0,
                'high_confidence_results': 0,
                'errors': 0
            },
            'generated_metadata': []
        }
        
        try:
            # Optimized: Query only selected objects instead of full catalog scan
            if selected_objects:
                selected_schema_names = selected_objects.get('schemas', [])
                selected_table_names = selected_objects.get('tables', [])
                selected_column_names = selected_objects.get('columns', [])
                
                logger.info(f"Exact selection scope - Schemas: {selected_schema_names}, Tables: {selected_table_names}, Columns: {selected_column_names}")
                
                # Query only the selected objects (much more efficient)
                missing_schemas = self._get_selected_schemas_metadata(catalog_name, selected_schema_names) if selected_schema_names else []
                missing_tables = self._get_selected_tables_metadata(catalog_name, selected_table_names) if selected_table_names else []
                missing_columns = self._get_selected_columns_metadata(catalog_name, selected_column_names) if selected_column_names else []
            else:
                # Fallback: Get all objects (only when no selection provided)
                logger.info("No selection provided, scanning entire catalog")
                missing_schemas = self.unity_service.get_schemas_with_missing_metadata(catalog_name)
                missing_tables = self.unity_service.get_tables_with_missing_metadata(catalog_name)
                missing_columns = self.unity_service.get_columns_with_missing_metadata(catalog_name)
            
            logger.info(f"Exact filtering result: {len(missing_schemas)} schemas, {len(missing_tables)} tables, {len(missing_columns)} columns")
            
            results['summary']['total_objects'] = len(missing_schemas) + len(missing_tables) + len(missing_columns)
            
            # Update progress: Setup complete
            self._update_progress(
                current_phase="Schema Analysis",
                phase_progress=0,
                processed_objects=0,
                current_object="Starting schema analysis..."
            )
            
            # Check for cancellation before starting schemas
            if self._is_cancelled(run_id):
                logger.info(f"🛑 Generation cancelled before schema processing")
                results['status'] = 'CANCELLED'
                results['cancelled_at'] = datetime.now().isoformat()
                return results
            
            # Process schemas in batches with intelligent sizing
            schema_batches = self._split_into_smart_batches(missing_schemas, 'schema')
            logger.info(f"📦 Processing {len(missing_schemas)} schemas in {len(schema_batches)} intelligent batches")
            
            # Define async batch processor with semaphore for parallel execution
            semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
            
            async def process_schema_batch_with_limit(batch_idx, schema_batch):
                async with semaphore:
                    # Check for cancellation before processing each batch
                    if self._is_cancelled(run_id):
                        logger.info(f"🛑 Skipping schema batch {batch_idx} due to cancellation")
                        return 0
                    
                    logger.info(f"📦 Processing schema batch {batch_idx}/{len(schema_batches)} ({len(schema_batch)} schemas)")
                    
                    try:
                        batch_metadata = await self._generate_schemas_batch(
                            schema_batch, catalog_name, model, run_id
                        )
                        
                        for metadata in batch_metadata:
                            results['generated_metadata'].append(metadata)
                            results['summary']['processed_objects'] += 1
                            
                            # Update progress after each schema
                            self._update_progress(
                                processed_objects=results['summary']['processed_objects'],
                                current_object=f"Schema: {metadata['full_name'].split('.')[1]}"
                            )
                            
                            if metadata.get('pii_detected'):
                                results['summary']['pii_objects_detected'] += 1
                            if metadata.get('confidence_score', 0) >= self.config['confidence_threshold']:
                                results['summary']['high_confidence_results'] += 1
                        
                        # CHECKPOINT: Save batch results incrementally (resume on failure support)
                        await self._save_checkpoint(batch_metadata, run_id)
                        
                        return len(batch_metadata)
                                
                    except Exception as e:
                        logger.error(f"Error processing schema batch: {e}")
                        results['summary']['errors'] += len(schema_batch)
                        # Update progress even on error
                        self._update_progress(
                            processed_objects=results['summary']['processed_objects'],
                            current_object=f"Schema batch (error)",
                            errors=[str(e)]
                        )
                        return 0
            
            # Execute all batches in parallel (with semaphore limiting to 5 concurrent)
            await asyncio.gather(*[
                process_schema_batch_with_limit(idx + 1, batch) 
                for idx, batch in enumerate(schema_batches)
            ])
            
            # Check for cancellation before starting tables
            if self._is_cancelled(run_id):
                logger.info(f"🛑 Generation cancelled after schema processing")
                results['status'] = 'CANCELLED'
                results['cancelled_at'] = datetime.now().isoformat()
                return results
            
            # Update progress: Moving to table analysis
            self._update_progress(
                current_phase="Table Analysis",
                phase_progress=0,
                current_object="Starting table analysis..."
            )
            
            # Process tables in batches with intelligent sizing
            table_batches = self._split_into_smart_batches(missing_tables, 'table')
            logger.info(f"📦 Processing {len(missing_tables)} tables in {len(table_batches)} intelligent batches")
            
            # Define async batch processor with semaphore for parallel execution
            semaphore_tables = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)
            
            async def process_table_batch_with_limit(batch_idx, table_batch):
                async with semaphore_tables:
                    # Check for cancellation before processing each batch
                    if self._is_cancelled(run_id):
                        logger.info(f"🛑 Skipping table batch {batch_idx} due to cancellation")
                        return 0
                    
                    logger.info(f"📦 Processing table batch {batch_idx}/{len(table_batches)} ({len(table_batch)} tables)")
                    
                    try:
                        batch_metadata = await self._generate_tables_batch(
                            table_batch, catalog_name, model, run_id
                        )
                        
                        for metadata in batch_metadata:
                            results['generated_metadata'].append(metadata)
                            results['summary']['processed_objects'] += 1
                            
                            # Update progress after each table
                            self._update_progress(
                                processed_objects=results['summary']['processed_objects'],
                                current_object=f"Table: {metadata['full_name'].split('.')[-1]}"
                            )
                            
                            if metadata.get('pii_detected'):
                                results['summary']['pii_objects_detected'] += 1
                            if metadata.get('confidence_score', 0) >= self.config['confidence_threshold']:
                                results['summary']['high_confidence_results'] += 1
                        
                        # CHECKPOINT: Save batch results incrementally (resume on failure support)
                        await self._save_checkpoint(batch_metadata, run_id)
                        
                        return len(batch_metadata)
                                
                    except Exception as e:
                        logger.error(f"Error processing table batch: {e}")
                        results['summary']['errors'] += len(table_batch)
                        # Update progress even on error
                        self._update_progress(
                            processed_objects=results['summary']['processed_objects'],
                            current_object=f"Table batch (error)"
                        )
                        return 0
            
            # Execute all batches in parallel (with semaphore limiting to 5 concurrent)
            await asyncio.gather(*[
                process_table_batch_with_limit(idx + 1, batch) 
                for idx, batch in enumerate(table_batches)
            ])
            
            # Check for cancellation before starting columns
            if self._is_cancelled(run_id):
                logger.info(f"🛑 Generation cancelled after table processing")
                results['status'] = 'CANCELLED'
                results['cancelled_at'] = datetime.now().isoformat()
                return results
            
            # Update progress: Moving to column analysis
            self._update_progress(
                current_phase="Column Analysis",
                phase_progress=0,
                current_object="Starting column analysis..."
            )
            
            # Process columns in optimized chunks
            column_results = await self._generate_columns_metadata_chunked(
                missing_columns, catalog_name, model, run_id
            )
            
            results['generated_metadata'].extend(column_results['metadata'])
            results['summary']['processed_objects'] += column_results['processed']
            results['summary']['pii_objects_detected'] += column_results['pii_detected']
            results['summary']['high_confidence_results'] += column_results['high_confidence']
            results['summary']['errors'] += column_results['errors']
            
            # Update progress: Finalization
            self._update_progress(
                current_phase="Finalization",
                phase_progress=100,
                processed_objects=results['summary']['processed_objects'],
                current_object="Completing generation..."
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced metadata generation: {e}")
            results['error'] = str(e)
            # Update progress on error
            self._update_progress(
                current_phase="Error",
                current_object=f"Generation failed: {str(e)}"
            )
        
        results['completed_at'] = datetime.now().isoformat()
        results['duration_seconds'] = (datetime.now() - start_time).total_seconds()
        
        # Final progress update
        self._update_progress(
            current_phase="Completed",
            phase_progress=100,
            processed_objects=results['summary']['processed_objects'],
            current_object="Generation completed successfully"
        )
        
        logger.info(f"Enhanced generation complete: {results['summary']}")
        return results
    
    async def _generate_schema_metadata_enhanced(self, schema_info: Dict, catalog_name: str, model: str, run_id: str = None) -> Dict:
        """Generate enhanced schema metadata with context and analysis"""
        schema_name = schema_info['name']
        
        # Get context: tables in this schema
        try:
            tables_response = await self._get_tables_in_schema(catalog_name, schema_name)
            table_names = [t['name'] for t in tables_response[:10]]  # Limit for context
            logger.info(f"🔍 Schema {schema_name}: Found {len(table_names)} tables: {table_names}")
        except Exception as e:
            logger.warning(f"Could not get tables for schema {schema_name}: {e}")
            table_names = []
        
        # Build enhanced context for LLM
        context_info = {
            'schema_name': schema_name,
            'catalog_name': catalog_name,
            'table_count': len(table_names),
            'table_names': table_names,
            'owner': schema_info.get('owner', ''),
            'created_at': schema_info.get('created_at', '')
        }
        
        # Generate description with context
        prompt = self._build_schema_prompt(context_info)
        
        try:
            description = self.llm_service._call_databricks_llm(
                prompt=prompt,
                max_tokens=self.config['max_tokens'],
                model=model,
                temperature=self.config['temperature']
            )
            
            # Debug: Log what we got from LLM service
            logger.info(f"Schema description from LLM service: {description[:200]}...")
            
            confidence_score = self._calculate_confidence_score(description, context_info, 'schema')
            
        except Exception as e:
            logger.error(f"LLM generation failed for schema {schema_name}: {e}")
            description = f"Schema containing {schema_name.replace('_', ' ')} related data and supporting tables"
            confidence_score = 0.3
        
        return {
            'run_id': run_id,
            'full_name': f"{catalog_name}.{schema_name}",
            'object_type': 'schema',
            'proposed_comment': description,
            'confidence_score': confidence_score,
            'pii_tags': [],
            'policy_tags': [],
            'data_classification': 'INTERNAL',
            'source_model': model,
            'generation_style': 'configured',  # Controlled via Prompts settings
            'context_used': context_info,
            'generated_at': datetime.now().isoformat(),
            'pii_detected': False
        }
    
    async def _generate_schemas_batch(self, schemas: List[Dict], catalog_name: str, model: str, run_id: str = None) -> List[Dict]:
        """Generate metadata for multiple schemas in a single LLM call (much more efficient)"""
        if not schemas:
            return []
        
        logger.info(f"📦 Batch generating descriptions for {len(schemas)} schemas")
        
        # Build context for each schema
        schema_contexts = []
        for schema_info in schemas:
            schema_name = schema_info['name']
            
            # Get table names for context
            try:
                tables_response = await self._get_tables_in_schema(catalog_name, schema_name)
                table_names = [t['name'] for t in tables_response[:5]]  # Limit to 5 for batch
            except:
                table_names = []
            
            schema_contexts.append({
                'schema_name': schema_name,
                'catalog_name': catalog_name,
                'table_count': len(table_names),
                'table_names': table_names
            })
        
        # For single schema, use the individual prompt builder directly (respects all custom settings)
        if len(schemas) == 1:
            batch_prompt = self._build_schema_prompt(schema_contexts[0])
        else:
            # For multiple schemas, build a batch prompt with custom settings
            batch_prompt = f"""Generate professional descriptions for these {len(schemas)} database schemas:\n\n"""
            
            for i, context in enumerate(schema_contexts, 1):
                table_context = f" | Tables: {', '.join(context['table_names'][:5])}" if context['table_names'] else ""
                batch_prompt += f"{i}. {context['schema_name']} ({context['table_count']} tables){table_context}\n"
            
            batch_prompt += "\n"
            
            # Get the configured prompt template (includes length, terminology, instructions)
            template_prompt = self._build_schema_prompt(schema_contexts[0])
            
            # Extract the requirements and formatting from the template
            # (everything after the "Context:" section)
            if "Requirements:" in template_prompt:
                requirements_section = template_prompt.split("Requirements:")[1]
                batch_prompt += "Requirements:" + requirements_section
        
        # Validate prompt size - if too large, recursively split
        if not self._validate_prompt_size(batch_prompt, len(schemas), 'schema'):
            if len(schemas) > 1:
                logger.warning(f"⚠️ Batch too large, splitting {len(schemas)} schemas into 2 sub-batches")
                mid = len(schemas) // 2
                results1 = await self._generate_schemas_batch(schemas[:mid], catalog_name, model, run_id)
                results2 = await self._generate_schemas_batch(schemas[mid:], catalog_name, model, run_id)
                return results1 + results2
            else:
                logger.error(f"❌ Single schema prompt too large, using fallback description")
                # Fallback for single item
                return [{
                    'run_id': run_id,
                    'full_name': f"{catalog_name}.{schemas[0]['name']}",
                    'object_type': 'schema',
                    'proposed_comment': f"Schema containing {schemas[0]['name'].replace('_', ' ')} related data",
                    'confidence_score': 0.3,
                    'pii_tags': [],
                    'policy_tags': [],
                    'proposed_policy_tags': [],
                    'data_classification': 'INTERNAL',
                    'source_model': model,
                    'generation_style': 'configured',  # Controlled via Prompts settings
                    'context_used': {'schema_name': schemas[0]['name']},
                    'pii_analysis': None,
                    'generated_at': datetime.now().isoformat(),
                    'pii_detected': False
                }]
        
        try:
            # Progress update: Calling LLM for batch
            self._update_progress(
                current_object=f"Generating descriptions for {len(schemas)} schemas..."
            )
            
            response = self.llm_service._call_databricks_llm(
                prompt=batch_prompt,
                max_tokens=1000,  # More tokens for batch
                model=model,
                temperature=self.config['temperature']
            )
            
            # Parse batch response into individual descriptions
            descriptions = self._parse_batch_descriptions(response, len(schemas))
            
        except Exception as e:
            logger.error(f"Batch LLM generation failed for schemas: {e}")
            # Fallback to simple descriptions
            descriptions = [
                f"Schema containing {schema['name'].replace('_', ' ')} related data and supporting tables"
                for schema in schemas
            ]
        
        # Build metadata results
        results = []
        for schema_info, description in zip(schemas, descriptions):
            schema_name = schema_info['name']
            confidence_score = self._calculate_confidence_score(description, {'schema_name': schema_name}, 'schema')
            
            results.append({
                'run_id': run_id,
                'full_name': f"{catalog_name}.{schema_name}",
                'object_type': 'schema',
                'proposed_comment': description,
                'confidence_score': confidence_score,
                'pii_tags': [],
                'policy_tags': [],
                'proposed_policy_tags': [],
                'data_classification': 'INTERNAL',
                'source_model': model,
                'generation_style': 'configured',  # Controlled via Prompts settings
                'context_used': {'schema_name': schema_name, 'catalog_name': catalog_name},
                'pii_analysis': None,
                'generated_at': datetime.now().isoformat(),
                'pii_detected': False
            })
        
        logger.info(f"✅ Batch generated {len(results)} schema descriptions")
        return results
    
    async def _generate_table_metadata_enhanced(self, table_info: Dict, catalog_name: str, model: str, run_id: str = None) -> Dict:
        """Generate enhanced table metadata using column names/types for context.
        
        NOTE: PII detection is NOT performed at table-level. 
        PII detection should ONLY happen when generating column-level metadata.
        """
        table_name = table_info['name']
        schema_name = table_info['schema_name']
        full_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        # Get column information (metadata only, no data sampling for table descriptions)
        try:
            columns_info = await self._get_table_columns_metadata(
                catalog_name, schema_name, table_name
            )
        except Exception as e:
            logger.warning(f"Could not get columns for table {full_name}: {e}")
            columns_info = []
        
        # NOTE: PII detection is NOT performed for table-level generation
        # PII detection should ONLY happen when generating column-level metadata
        # Here we just use column names/types for context to help LLM understand table structure
        
        # Build enhanced context with optional sample values
        context_info = {
            'table_name': table_name,
            'schema_name': schema_name,
            'catalog_name': catalog_name,
            'table_type': table_info.get('table_type', ''),
            'column_count': len(columns_info),
            'column_names': [col['name'] for col in columns_info[:15]],  # Limit for context
            'data_types': [col['data_type'] for col in columns_info[:15]],
            'owner': table_info.get('owner', ''),
            'pii_analysis': None  # No PII analysis for tables
        }
        
        # Add sample values if sampling is enabled (no PII redaction needed for tables)
        if self.config.get('enable_sampling', True) and columns_info:
            columns_with_samples = self._prepare_column_samples_for_context(
                columns_info[:15],  # Limit to first 15 columns
                None,  # No PII analysis for tables
                redact_pii=False  # No PII redaction for table-level generation
            )
            context_info['column_samples'] = columns_with_samples
            logger.info(f"📊 Including sample values for {len(columns_with_samples)} columns in LLM context")
        
        # Generate description with enhanced context
        prompt = self._build_table_prompt(context_info)
        
        try:
            description = self.llm_service._call_databricks_llm(
                prompt=prompt,
                max_tokens=self.config['max_tokens'],
                model=model,
                temperature=self.config['temperature']
            )
            
            confidence_score = self._calculate_confidence_score(description, context_info, 'table')
            
        except Exception as e:
            logger.error(f"LLM generation failed for table {full_name}: {e}")
            description = f"Table containing {table_name.replace('_', ' ')} data with related attributes"
            confidence_score = 0.3
        
        # Tables do not have PII detection - PII detection is only for columns
        pii_detected = False
        pii_tags = []
        policy_tags = []
        classification = 'INTERNAL'
        
        return {
            'run_id': run_id,
            'full_name': full_name,
            'object_type': 'table',
            'proposed_comment': description,
            'confidence_score': confidence_score,
            'pii_tags': json.dumps([]),  # Empty for tables
            'policy_tags': json.dumps([]),  # Empty for tables
            'data_classification': classification,
            'source_model': model,
            'generation_style': 'configured',  # Controlled via Prompts settings
            'context_used': context_info,
            'pii_analysis': None,  # No PII analysis for tables
            'generated_at': datetime.now().isoformat(),
            'pii_detected': False  # Always False for tables
        }
    
    async def _generate_tables_batch(self, tables: List[Dict], catalog_name: str, model: str, run_id: str = None) -> List[Dict]:
        """Generate metadata for multiple tables in a single LLM call (much more efficient)"""
        if not tables:
            return []
        
        logger.info(f"📦 Batch generating descriptions for {len(tables)} tables")
        
        # Collect column info and run PII detection for all tables
        table_contexts = []
        for idx, table_info in enumerate(tables, 1):
            table_name = table_info['name']
            schema_name = table_info['schema_name']
            full_name = f"{catalog_name}.{schema_name}.{table_name}"
            
            # Progress update: Analyzing table
            self._update_progress(
                current_object=f"{table_name} ({idx}/{len(tables)} in batch)",
                phase_progress=int((idx / len(tables)) * 100)
            )
            
            # Get column information (metadata only, no data sampling for table descriptions)
            try:
                columns_info = await self._get_table_columns_metadata(
                    catalog_name, schema_name, table_name
                )
            except Exception as e:
                logger.warning(f"Could not get columns for table {full_name}: {e}")
                columns_info = []
            
            # NOTE: No PII analysis for table-level generation
            # PII detection should ONLY happen when generating column-level metadata
            
            # Build context with optional sample values
            context = {
                'table_name': table_name,
                'schema_name': schema_name,
                'catalog_name': catalog_name,
                'table_type': table_info.get('table_type', ''),
                'column_count': len(columns_info),
                'column_names': [col['name'] for col in columns_info[:10]],  # Limit to 10 for batch
                'data_types': list(set([col['data_type'] for col in columns_info[:10]])),
                'owner': table_info.get('owner', ''),
                'pii_analysis': None,  # No PII analysis for tables
                'columns_info': columns_info
            }
            
            # Add sample values if sampling is enabled (no PII redaction for tables)
            if self.config.get('enable_sampling', True) and columns_info:
                columns_with_samples = self._prepare_column_samples_for_context(
                    columns_info[:8],  # Limit to 8 columns for batch
                    None,  # No PII analysis for tables
                    redact_pii=False  # No PII redaction for table-level generation
                )
                context['column_samples'] = columns_with_samples
            
            table_contexts.append(context)
        
        # For single table, use the individual prompt builder directly (respects all custom settings)
        if len(tables) == 1:
            batch_prompt = self._build_table_prompt(table_contexts[0])
        else:
            # For multiple tables, build a batch prompt with custom settings
            batch_prompt = f"""Generate professional descriptions for these {len(tables)} database tables:\n\n"""
            
            for i, ctx in enumerate(table_contexts, 1):
                col_names_str = ', '.join(ctx['column_names'][:5])
                
                sample_info = ""
                if ctx.get('column_samples'):
                    sample_preview = []
                    for col_sample in ctx['column_samples'][:3]:  # Show 3 sample columns
                        samples_str = ', '.join([f"'{v}'" for v in col_sample['sample_values'][:2]])
                        sample_preview.append(f"{col_sample['column_name']}: {samples_str}")
                    sample_info = f" | Samples: {'; '.join(sample_preview)}"
                
                batch_prompt += f"{i}. {ctx['schema_name']}.{ctx['table_name']} ({ctx['column_count']} columns: {col_names_str}){sample_info}\n"
            
            batch_prompt += "\n"
            
            # Get the configured prompt template (includes length, terminology, instructions)
            template_prompt = self._build_table_prompt(table_contexts[0])
            
            # Extract the requirements and formatting from the template
            # (everything after the "Context:" section)
            if "Requirements:" in template_prompt:
                requirements_section = template_prompt.split("Requirements:")[1]
                batch_prompt += "Requirements:" + requirements_section
            
            # ADD CRITICAL BATCH-SPECIFIC FORMAT INSTRUCTIONS
            # This is essential for robust parsing across different LLM models
            batch_prompt += f"""

🎯 CRITICAL OUTPUT FORMAT FOR BATCH RESPONSES:
YOU MUST return exactly {len(tables)} descriptions in this EXACT format:

1. Description for first table
2. Description for second table
3. Description for third table
...and so on

STRICT RULES:
- Start each line with the number followed by a period and space: "1. ", "2. ", "3. "
- Do NOT include table names in the response
- Do NOT use bold/italic markers like **text** or *text*
- Do NOT use headers or section dividers
- Each description should be plain text on its own line
- Descriptions can span multiple lines, but must start with the number
- Do NOT add extra blank lines between descriptions

EXAMPLE (for 3 tables):
1. Stores customer transaction data including purchase amounts and dates, supporting financial reporting and revenue analysis.
2. Contains product catalog information with pricing and inventory levels for e-commerce operations.
3. Maintains employee records and organizational hierarchy for HR management and reporting purposes.

Return exactly {len(tables)} numbered descriptions in the format above. This format is MANDATORY for proper parsing."""
        
        # Validate prompt size - if too large, recursively split
        if not self._validate_prompt_size(batch_prompt, len(tables), 'table'):
            if len(tables) > 1:
                logger.warning(f"⚠️ Batch too large, splitting {len(tables)} tables into 2 sub-batches")
                mid = len(tables) // 2
                results1 = await self._generate_tables_batch(tables[:mid], catalog_name, model, run_id)
                results2 = await self._generate_tables_batch(tables[mid:], catalog_name, model, run_id)
                return results1 + results2
            else:
                logger.error(f"❌ Single table prompt too large, using fallback description")
                # Fallback for single complex table
                ctx = table_contexts[0]
                return [{
                    'run_id': run_id,
                    'full_name': f"{ctx['catalog_name']}.{ctx['schema_name']}.{ctx['table_name']}",
                    'object_type': 'table',
                    'proposed_comment': f"Table containing {ctx['table_name'].replace('_', ' ')} information",
                    'confidence_score': 0.3,
                    'pii_tags': '[]',  # Empty for tables
                    'policy_tags': '[]',  # Empty for tables
                    'proposed_policy_tags': '[]',  # Empty for tables
                    'data_classification': 'INTERNAL',  # Default for tables
                    'source_model': model,
                    'generation_style': 'configured',  # Controlled via Prompts settings
                    'context_used': ctx,
                    'pii_analysis': None,  # No PII analysis for tables
                    'generated_at': datetime.now().isoformat(),
                    'pii_detected': False  # Always False for tables
                }]
        
        try:
            # Progress update: Calling LLM for batch
            self._update_progress(
                current_object=f"Generating descriptions for {len(tables)} tables..."
            )
            
            response = self.llm_service._call_databricks_llm(
                prompt=batch_prompt,
                max_tokens=1500,  # More tokens for table batch
                model=model,
                temperature=self.config['temperature']
            )
            
            # Parse batch response
            descriptions = self._parse_batch_descriptions(response, len(tables))
            
        except Exception as e:
            logger.error(f"Batch LLM generation failed for tables: {e}")
            # Fallback to simple descriptions
            descriptions = [
                f"Table containing {ctx['table_name'].replace('_', ' ')} information with {ctx['column_count']} columns"
                for ctx in table_contexts
            ]
        
        # Build metadata results
        results = []
        for ctx, description in zip(table_contexts, descriptions):
            full_name = f"{ctx['catalog_name']}.{ctx['schema_name']}.{ctx['table_name']}"
            
            # Tables do not have PII detection - PII detection is only for columns
            confidence_score = self._calculate_confidence_score(description, ctx, 'table', None)
            
            results.append({
                'run_id': run_id,
                'full_name': full_name,
                'object_type': 'table',
                'proposed_comment': description,
                'confidence_score': confidence_score,
                'pii_tags': '[]',  # Empty for tables
                'policy_tags': '[]',  # Empty for tables
                'proposed_policy_tags': '[]',  # Empty for tables
                'data_classification': 'INTERNAL',  # Default for tables
                'source_model': model,
                'generation_style': 'configured',  # Controlled via Prompts settings
                'context_used': ctx,
                'pii_analysis': None,  # No PII analysis for tables
                'generated_at': datetime.now().isoformat(),
                'pii_detected': False  # Always False for tables
            })
        
        logger.info(f"✅ Batch generated {len(results)} table descriptions")
        return results
    
    async def _generate_columns_metadata_chunked(self, columns: List[Dict], catalog_name: str, model: str, run_id: str = None) -> Dict:
        """Generate column metadata using intelligent chunking for efficiency"""
        results = {
            'metadata': [],
            'processed': 0,
            'pii_detected': 0,
            'high_confidence': 0,
            'errors': 0
        }
        
        if not columns:
            return results
        
        # Group columns by table for context efficiency
        table_groups = {}
        for column in columns:
            table_key = f"{column['catalog_name']}.{column['schema_name']}.{column['table_name']}"
            if table_key not in table_groups:
                table_groups[table_key] = []
            table_groups[table_key].append(column)
        
        # Process each table's columns together
        for table_key, table_columns in table_groups.items():
            # Check for cancellation before processing each table's columns
            if self._is_cancelled(run_id):
                logger.info(f"🛑 Stopping column processing for {table_key} due to cancellation")
                break
            
            try:
                # OPTIMIZATION: Sample the table ONCE for all columns, then reuse for each chunk
                # This dramatically reduces database queries for wide tables (50+ columns)
                table_sample_cache = None
                if table_columns:
                    first_col = table_columns[0]
                    catalog_name = first_col['catalog_name']
                    schema_name = first_col['schema_name']
                    table_name = first_col['table_name']
                    
                    # Sample ALL columns from this table once
                    all_column_samples = await self._get_table_columns_with_samples(
                        catalog_name, schema_name, table_name
                    )
                    
                    # Convert to a dict for fast lookup by column name
                    table_sample_cache = {col['name']: col for col in all_column_samples}
                    logger.info(f"📦 Cached sample data for {len(table_sample_cache)} columns in {table_key}")
                
                # OPTIMIZATION: For wide tables, create chunks first, then sample per-chunk
                # This prevents sampling 1000+ columns upfront for very wide tables
                chunks = self._create_optimal_chunks(table_columns)
                
                for chunk in chunks:
                    # Check for cancellation before processing each chunk
                    if self._is_cancelled(run_id):
                        logger.info(f"🛑 Stopping chunk processing due to cancellation")
                        break
                    
                    try:
                        chunk_results = await self._generate_column_chunk_metadata(
                            chunk, model, run_id, table_sample_cache=table_sample_cache
                        )
                        
                        results['metadata'].extend(chunk_results['metadata'])
                        results['processed'] += chunk_results['processed']
                        results['pii_detected'] += chunk_results['pii_detected']
                        results['high_confidence'] += chunk_results['high_confidence']
                        
                        # CHECKPOINT: Save chunk results incrementally (resume on failure support)
                        await self._save_checkpoint(chunk_results['metadata'], run_id)
                        
                        # Batch progress updates: Only update every 5 chunks or for large chunks to reduce overhead
                        if len(chunk) >= 5 or results['processed'] % 50 == 0:
                            self._update_progress(
                                processed_objects=results['processed'],
                                current_object=f"Columns in {table_key} (chunk {len(chunk)} columns)"
                            )
                        
                    except Exception as e:
                        logger.error(f"Error processing column chunk in {table_key}: {e}")
                        results['errors'] += len(chunk)
                        # Update progress on error
                        self._update_progress(
                            processed_objects=results['processed'],
                            current_object=f"Columns in {table_key} (error)"
                        )
                        
            except Exception as e:
                logger.error(f"Error processing columns for table {table_key}: {e}")
                results['errors'] += len(table_columns)
        
        return results
    
    async def _generate_column_chunk_metadata(self, columns: List[Dict], model: str, run_id: str = None, is_improvement: bool = False, table_sample_cache: Dict = None) -> Dict:
        """Generate metadata for a chunk of columns with shared context
        
        Args:
            columns: List of column dicts
            model: LLM model to use
            run_id: Optional run ID for progress tracking
            is_improvement: If True, uses improvement prompt instead of generation prompt
            table_sample_cache: Optional pre-cached sample data for the table to avoid re-querying
        """
        if not columns:
            return {'metadata': [], 'processed': 0, 'pii_detected': 0, 'high_confidence': 0}
        
        # OPTIMIZATION: Sample data for this specific chunk only (not entire table)
        # This makes processing 1000+ column tables much more efficient
        # If table_sample_cache is provided, use it instead of re-querying
        enriched_columns = await self._enrich_columns_with_samples(columns, table_sample_cache=table_sample_cache)
        
        # All columns in chunk are from same table
        first_col = enriched_columns[0]
        table_context = {
            'catalog_name': first_col['catalog_name'],
            'schema_name': first_col['schema_name'],
            'table_name': first_col['table_name'],
            'columns': enriched_columns
        }
        
        results = {
            'metadata': [],
            'processed': 0,
            'pii_detected': 0,
            'high_confidence': 0
        }
        
        # Build appropriate prompt based on mode
        if is_improvement:
            prompt = self._build_improvement_prompt(table_context)
        else:
            prompt = self._build_column_batch_prompt(table_context)
        
        # Log sample data availability
        columns_with_samples = sum(1 for col in columns if col.get('sample_values'))
        redaction_enabled = self.config.get('redact_pii_in_samples', False)
        
        if columns_with_samples > 0:
            if redaction_enabled:
                logger.info(f"📊 Column batch prompt includes sample data for {columns_with_samples}/{len(columns)} columns (🔒 REDACTED for privacy)")
            else:
                logger.info(f"📊 Column batch prompt includes sample data for {columns_with_samples}/{len(columns)} columns")
        else:
            # Provide context-specific message based on whether sampling is enabled
            if not self.config.get('enable_sampling', True):
                logger.info(f"📊 Column batch prompt without sample data (sampling disabled in settings)")
            else:
                logger.warning(f"⚠️ Column batch prompt has NO sample data for {len(columns)} columns (sampling may have failed)")
        
        try:
            # Get batch response from LLM
            batch_response = self.llm_service._call_databricks_llm(
                prompt=prompt,
                max_tokens=self.config['max_tokens'] * len(columns),
                model=model,
                temperature=self.config['temperature']
            )
            
            # Parse batch response into individual descriptions
            descriptions = self._parse_batch_response(batch_response, columns)
            
        except Exception as e:
            logger.error(f"Batch LLM generation failed: {e}")
            # Fallback to simple descriptions
            descriptions = [
                f"Contains {col['column_name'].replace('_', ' ')} information"
                for col in columns
            ]
        
        # Add descriptions to column dicts BEFORE PII detection (so PII detector can use them)
        for i, col in enumerate(columns):
            if i < len(descriptions):
                col['description'] = descriptions[i]
        
        logger.debug(f"📝 Enriched {len(columns)} columns with generated descriptions for PII analysis")
        
        # Batch PII analysis for all columns (much more efficient than per-column calls)
        pii_enabled = self.config['enable_pii_detection']
        if self.settings_manager and pii_enabled:
            try:
                pii_config = self.settings_manager.get_pii_config()
                pii_enabled = pii_config.get('enabled', True)
            except Exception as e:
                logger.warning(f"Failed to get PII settings, using config default: {e}")
        
        pii_analyses_by_column = {}
        if pii_enabled:
            # Check PII cache first (PII is data-based, shared across models)
            columns_needing_pii = []
            for col in columns:
                cache_key = f"{col['catalog_name']}.{col['schema_name']}.{col['table_name']}.{col['column_name']}"
                if cache_key in self.pii_cache:
                    # Cache hit! Reuse PII result from previous model
                    pii_analyses_by_column[col['column_name']] = self.pii_cache[cache_key]
                    logger.debug(f"♻️ PII cache hit for {col['column_name']}")
                else:
                    # Cache miss - need to analyze this column
                    columns_needing_pii.append(col)
            
            # Only call PII detector for columns not in cache
            if columns_needing_pii:
                logger.info(f"🔒 Analyzing {len(columns_needing_pii)} new columns for PII (cached: {len(columns) - len(columns_needing_pii)})")
                pii_model_to_use = getattr(self, '_current_pii_model', None)
                pii_results = self.pii_detector.analyze_columns_batch(columns_needing_pii, llm_model=pii_model_to_use)
                
                # Store results in both local dict and persistent cache
                for col, pii_result in zip(columns_needing_pii, pii_results):
                    cache_key = f"{col['catalog_name']}.{col['schema_name']}.{col['table_name']}.{col['column_name']}"
                    pii_analyses_by_column[col['column_name']] = pii_result
                    self.pii_cache[cache_key] = pii_result  # Cache for next model
            else:
                logger.info(f"♻️ All {len(columns)} columns found in PII cache - skipping detection")
        
        # Process each column
        for i, column in enumerate(columns):
            try:
                description = descriptions[i] if i < len(descriptions) else descriptions[0]
                
                # Get PII analysis from batch results
                pii_analysis = pii_analyses_by_column.get(column['column_name'])
                
                confidence_score = self._calculate_confidence_score(
                    description, column, 'column', pii_analysis
                )
                
                # Build result
                full_name = f"{column['catalog_name']}.{column['schema_name']}.{column['table_name']}.{column['column_name']}"
                
                # Only flag as PII detected if classification is truly sensitive (not PUBLIC or INTERNAL)
                classification = pii_analysis['classification'] if pii_analysis else 'INTERNAL'
                pii_detected = classification in ['PII', 'PHI', 'PCI', 'CONFIDENTIAL', 'SENSITIVE']
                pii_tags = pii_analysis['pii_types'] if pii_analysis else []
                # Use proposed tags instead of automatic tags
                proposed_policy_tags = pii_analysis['proposed_policy_tags'] if pii_analysis else []
                policy_tags = []  # No automatic tags - only proposed ones
                
                # Log PII detection for debugging
                if pii_detected:
                    tag_status = f"{len(proposed_policy_tags)} tags proposed" if proposed_policy_tags else "no tags proposed (may conflict with governed tags)"
                    logger.debug(f"🔒 PII detected for {column['column_name']}: classification={classification}, types={pii_tags}, {tag_status}")
                
                metadata = {
                    'run_id': run_id,
                    'full_name': full_name,
                    'object_type': 'column',
                    'proposed_comment': description,
                    'confidence_score': confidence_score,
                    'pii_tags': json.dumps(pii_tags),
                    'policy_tags': json.dumps(policy_tags),  # Empty - no automatic tags
                    'proposed_policy_tags': json.dumps(proposed_policy_tags),  # New - for manual review
                    'data_classification': classification,
                    'source_model': model,
                    'generation_style': 'configured',  # Controlled via Prompts settings
                    'context_used': table_context,
                    'pii_analysis': pii_analysis,
                    'generated_at': datetime.now().isoformat(),
                    'pii_detected': pii_detected
                }
                
                results['metadata'].append(metadata)
                results['processed'] += 1
                
                if pii_detected:
                    results['pii_detected'] += 1
                if confidence_score >= self.config['confidence_threshold']:
                    results['high_confidence'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing column {column['column_name']}: {e}")
                continue
        
        return results
    
    def _build_schema_prompt(self, context: Dict) -> str:
        """Build prompt for schema description using configured length"""
        schema_name = context['schema_name']
        table_names = context.get('table_names', [])
        table_count = context.get('table_count', 0)
        
        # Get length setting for schemas
        prompt_config = self._get_prompt_config()
        length_config = prompt_config.get('description_length', {})
        if isinstance(length_config, dict):
            schema_length = length_config.get('schema', 'detailed')
        else:
            schema_length = length_config  # Legacy
        
        length_guidance = {
            'concise': "1-2 sentences, brief and direct",
            'standard': "2-3 sentences with context",
            'detailed': "3-4 sentences with comprehensive business context and domain significance"
        }
        
        logger.info(f"📏 Schema length setting: '{schema_length}' -> {length_guidance.get(schema_length, length_guidance['detailed'])}")
        
        # Build table context for business purpose inference
        if table_names:
            table_context = f"- Contains {table_count} tables: {', '.join(table_names[:5])}"
            if len(table_names) > 5:
                table_context += f" (and {len(table_names) - 5} more)"
        else:
            table_context = f"- Contains {table_count} tables (names not available)"
        
        base_prompt = f"""
Generate a description for the database schema '{schema_name}' in {length_guidance.get(schema_length, length_guidance['detailed'])}.

Context:
- Schema: {schema_name}
{table_context}

Requirements:
- Maximum 4 sentences
- INFER the business domain and purpose from the schema name and table names
- DO NOT mention table names, counts, or list tables in the description
- Use the table names as intelligence to understand the business domain
- Explain what business processes and data types this schema supports
- Professional, clear tone
- NO tables, bullet points, or formatting
- Plain text only

Business Domain Inference Examples:
- Tables like "patient", "diagnosis", "treatment" → Focus on healthcare data management and clinical operations
- Tables like "claims", "policies", "coverage" → Focus on insurance operations and risk management
- Tables like "orders", "customers", "products" → Focus on e-commerce and retail operations
- Tables like "transactions", "accounts", "balances" → Focus on financial services and banking

Generate a business-focused description that explains the domain purpose without mentioning specific tables.

OUTPUT FORMAT:
Return ONLY the description as plain text, starting directly with the content. Do NOT number it, do NOT add prefixes like "Description:" or preamble text.
Example: "The schema serves as a centralized repository for customer relationship data, supporting sales operations and customer service workflows across the organization..."
"""
        
        # Add custom terminology if defined
        custom_terminology = prompt_config.get('custom_terminology', {})
        if custom_terminology:
            base_prompt += "\n\nIMPORTANT TERMINOLOGY:"
            for term, meaning in custom_terminology.items():
                base_prompt += f"\n- '{term}' means '{meaning}'"
            logger.info(f"🔤 Added {len(custom_terminology)} custom terminology mappings to schema prompt")
        
        # Add additional custom instructions if provided
        additional_instructions = prompt_config.get('additional_instructions', '').strip()
        if additional_instructions:
            base_prompt += f"\n\nADDITIONAL REQUIREMENTS:\n{additional_instructions}"
            logger.info(f"📝 Added custom instructions to schema prompt: {additional_instructions[:100]}{'...' if len(additional_instructions) > 100 else ''}")
        
        return base_prompt.strip()
    
    def _sanitize_description(self, description: str) -> str:
        """
        Remove PII-related language from descriptions for PUBLIC/INTERNAL data.
        
        LLMs sometimes hallucinate about data sensitivity despite explicit prohibitions.
        This post-processing ensures clean, business-focused descriptions.
        """
        import re
        
        # Patterns to remove (case-insensitive)
        # Match full sentences that mention PII/security/compliance
        pii_patterns = [
            r'[^.]*[Cc]ontains\s+personally\s+identifiable\s+information[^.]*\.',
            r'[^.]*[Pp]ersonally\s+identifiable\s+information\s+\(PII\)[^.]*\.',
            r'[^.]*[Rr]equires?\s+appropriate\s+access\s+controls[^.]*\.',
            r'[^.]*[Rr]equires?\s+data\s+handling\s+procedures[^.]*\.',
            r'[^.]*[Aa]ppropriate\s+access\s+controls\s+and\s+data\s+handling\s+procedures[^.]*\.',
            r'[^.]*[Ss]ensitive\s+data[^.]*\s+protection[^.]*\.',
            r'[^.]*[Dd]ata\s+security\s+measures[^.]*\.',
            r'[^.]*[Cc]ompliance\s+requirements[^.]*\.',
        ]
        
        cleaned = description
        removed_patterns = []
        
        for i, pattern in enumerate(pii_patterns):
            before = cleaned
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            if before != cleaned:
                removed_patterns.append(f"pattern_{i+1}")
        
        # Clean up extra spaces and periods
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces → single space
        cleaned = re.sub(r'\s+\.', '.', cleaned)  # Space before period
        cleaned = re.sub(r'\.+', '.', cleaned)  # Multiple periods → single
        cleaned = cleaned.strip()
        
        # Log what was removed (only if something changed)
        if cleaned != description:
            logger.info(f"🧹 Sanitized {len(removed_patterns)} PII hallucination(s) from description")
        
        return cleaned
    
    def _build_table_prompt(self, context: Dict) -> str:
        """Build prompt for table description using configured length"""
        table_name = context['table_name']
        column_names = context.get('column_names', [])
        pii_analysis = context.get('pii_analysis')
        column_samples = context.get('column_samples', [])
        
        # Get length setting for tables
        prompt_config = self._get_prompt_config()
        length_config = prompt_config.get('description_length', {})
        if isinstance(length_config, dict):
            table_length = length_config.get('table', 'standard')
        else:
            table_length = length_config  # Legacy
        
        length_guidance = {
            'concise': "1-2 sentences, brief and direct",
            'standard': "2-3 sentences with context",
            'detailed': "3-4 sentences with comprehensive business context and domain significance"
        }
        
        logger.info(f"📏 Table length setting: '{table_length}' -> {length_guidance.get(table_length, length_guidance['standard'])}")
        
        # Build column context for business purpose inference
        if column_names:
            column_context = f"- Contains {len(column_names)} columns: {', '.join(column_names[:15])}"
            if len(column_names) > 15:
                column_context += f" (and {len(column_names) - 15} more)"
        else:
            column_context = f"- Contains {len(column_names)} columns (names not available)"
        
        base_prompt = f"""
Generate a description for the table '{table_name}' in {length_guidance.get(table_length, length_guidance['standard'])}.

Context:
- Table: {table_name}
{column_context}
- Data types: {', '.join(set(context.get('data_types', [])))}
"""
        
        # Add sample values if available
        if column_samples:
            base_prompt += "\n- Sample data patterns:\n"
            for col_sample in column_samples[:8]:  # Show up to 8 columns with samples
                samples_str = ', '.join([f"'{v}'" for v in col_sample['sample_values'][:3]])  # First 3 samples
                base_prompt += f"  • {col_sample['column_name']} ({col_sample['data_type']}): {samples_str}\n"
            if len(column_samples) > 8:
                base_prompt += f"  (and {len(column_samples) - 8} more columns with samples)\n"
        
        # Only include PII info if it's truly sensitive (PHI, PCI, HIGH risk)
        # DO NOT include for generic location data or low-risk classifications
        if pii_analysis and pii_analysis.get('highest_classification') in ['PHI', 'PCI'] and pii_analysis.get('risk_assessment') in ['HIGH', 'CRITICAL']:
            base_prompt += f"""
- Note: Contains regulated data (classification: {pii_analysis['highest_classification']})
"""
        
        base_prompt += f"""
Requirements:
- Maximum 3 sentences
- INFER the business purpose from the table name, column names, and sample data patterns
- Focus ONLY on what the data represents and its business use case
- DO NOT mention column names, counts, or list columns in the description
- Use the column names and sample values as intelligence to understand the data purpose
- Explain what business entity or process this table represents
- Professional, clear tone
- NO formatting, tables, or bullet points
- Plain text only

CRITICAL PROHIBITIONS:
- NEVER mention "personally identifiable information", "PII", "sensitive data", or "access controls"
- NEVER mention "data handling procedures", "security", or "compliance" 
- NEVER add disclaimers about data sensitivity or protection requirements
- These are business metadata descriptions, NOT security assessments

Business Purpose Inference Examples:
- Weather/forecast data → "Stores weather predictions for analysis and planning"
- Transaction data → "Captures business transactions for revenue tracking" 
- Customer data → "Maintains customer information for service delivery"
- Product data → "Contains product catalog for inventory management"

Generate a business-focused description that explains what the data is used for.

OUTPUT FORMAT:
Return ONLY the description as plain text, starting directly with the content. Do NOT number it, do NOT add prefixes like "Description:" or preamble text.
Example: "The table stores customer transaction data including purchase amounts and dates, supporting financial reporting and revenue analysis processes..."
"""
        
        # Add custom terminology if defined
        custom_terminology = prompt_config.get('custom_terminology', {})
        if custom_terminology:
            base_prompt += "\n\nIMPORTANT TERMINOLOGY:"
            for term, meaning in custom_terminology.items():
                base_prompt += f"\n- '{term}' means '{meaning}'"
            logger.info(f"🔤 Added {len(custom_terminology)} custom terminology mappings to table prompt")
        
        # Add additional custom instructions if provided
        additional_instructions = prompt_config.get('additional_instructions', '').strip()
        if additional_instructions:
            base_prompt += f"\n\nADDITIONAL REQUIREMENTS:\n{additional_instructions}"
            logger.info(f"📝 Added custom instructions to table prompt: {additional_instructions[:100]}{'...' if len(additional_instructions) > 100 else ''}")
        
        return base_prompt.strip()
    
    def _get_prompt_config(self) -> dict:
        """Get prompt configuration from settings"""
        try:
            if self.settings_manager:
                settings = self.settings_manager.get_settings()
                if settings and 'prompt_config' in settings:
                    config = settings['prompt_config']
                    # Debug logging to show what configuration is being used
                    logger.info(f"🎯 Using custom prompt config: length={config.get('description_length')}, terminology={len(config.get('custom_terminology', {}))} terms, instructions={'Yes' if config.get('additional_instructions') else 'No'}")
                    if config.get('custom_terminology'):
                        logger.info(f"🔤 Custom terminology: {config['custom_terminology']}")
                    return config
        except Exception as e:
            logger.warning(f"Failed to load prompt config, using defaults: {e}")
        
        # Return defaults if loading fails
        logger.info("🎯 Using default prompt config (no custom settings found)")
        return {
            'custom_terminology': {},
            'additional_instructions': '',
            'description_length': {
                'schema': 'detailed',
                'table': 'standard',
                'column': 'concise'
            },
            'include_technical_details': True,
            'include_business_context': True,
            'custom_examples': []
        }
    
    def _build_improvement_prompt(self, context: Dict) -> str:
        """Build prompt specifically for improving existing descriptions"""
        table_name = context['table_name']
        columns = context['columns']
        
        prompt = f"""
TASK: Improve the existing descriptions for columns in table '{table_name}'.

CRITICAL EVALUATION REQUIRED:
First, evaluate the quality of each current description:

REPLACE COMPLETELY if the description is:
- Nonsensical or placeholder text (e.g., "my name is jim", "test data", "TODO", "N/A")
- Personal comments or unrelated content (e.g., "john was here", "for bob's team")
- Generic/meaningless (e.g., "data field", "column", "information")
- Completely wrong for the column name and data type shown
- Empty, minimal, or unhelpful (e.g., "id", "name field")

ENHANCE AND PRESERVE if the description:
- Contains valuable business context and organizational significance
- Shows understanding of the data's purpose and usage
- Is mostly correct but could be clearer or more concise
- Has good business meaning but awkward phrasing

When in doubt: If the description seems professional and relevant, improve it. If it seems like test/placeholder/nonsense, replace it entirely.

"""
        
        for i, col in enumerate(columns, 1):
            current_desc = col.get('comment', '').strip()
            sample_values = col.get('sample_values', [])
            
            # Format sample values
            if sample_values:
                samples_preview = []
                for v in sample_values[:3]:
                    if v is None:
                        samples_preview.append('NULL')
                    else:
                        v_str = str(v)
                        if len(v_str) > 50:
                            v_str = v_str[:47] + '...'
                        samples_preview.append(f"'{v_str}'")
                samples_str = ', '.join(samples_preview)
            else:
                samples_str = '(no samples available)'
            
            prompt += f"""
{i}. Column: {col['column_name']}
   Type: {col['data_type']}
   Sample values: {samples_str}
   CURRENT DESCRIPTION: "{current_desc}"

"""
        
        prompt_config = self._get_prompt_config()
        length_config = prompt_config.get('description_length', 'standard')
        if isinstance(length_config, dict):
            description_length = length_config.get('column', 'concise')
        else:
            description_length = length_config
        
        length_guidance = {
            'concise': "Keep improved descriptions brief and direct (1 short sentence, ~15-25 words).",
            'standard': "Provide clear, professional improvements (1-2 sentences, ~30-50 words).",
            'detailed': "Generate comprehensive improvements with business context (2-3 sentences, ~60-100 words)."
        }
        
        prompt += f"\nIMPROVEMENT GUIDELINES:"
        prompt += f"\n- For BAD descriptions: Generate a completely new, professional description from scratch based on column name, data type, and sample values"
        prompt += f"\n- For GOOD descriptions: Preserve valuable business context and enhance clarity/precision"
        prompt += f"\n- Add insights from sample data without mentioning specific values"
        prompt += f"\n- Remove redundancy, awkward phrasing, or technical jargon"
        prompt += f"\n- If current description is excellent, make only minor improvements"
        prompt += f"\n- Focus on business value and organizational significance"
        prompt += f"\n\nLength: {length_guidance.get(description_length, length_guidance['standard'])}"
        
        prompt += f"\n\nEXAMPLES:"
        prompt += f"\n"
        prompt += f"\nExample 1 - BAD description to REPLACE:"
        prompt += f"\n  Column: customer_id | Type: STRING | Samples: 'C12345', 'C67890'"
        prompt += f"\n  Current: 'my name is jim'"
        prompt += f"\n  Improved: 'Unique identifier for customers, used to link transactions and track customer relationships across systems.'"
        prompt += f"\n"
        prompt += f"\nExample 2 - GOOD description to ENHANCE:"
        prompt += f"\n  Column: order_date | Type: DATE | Samples: '2024-01-15', '2024-01-16'"
        prompt += f"\n  Current: 'This column stores the date when customer places order, enabling time-based analysis for business intelligence reporting.'"
        prompt += f"\n  Improved: 'Records the date when a customer places an order, enabling time-series analysis and business intelligence reporting.'"
        prompt += f"\n"
        prompt += f"\n\nOUTPUT FORMAT (CRITICAL - FOLLOW EXACTLY):"
        prompt += f"\nReturn exactly {len(columns)} improved descriptions, numbered as shown below."
        prompt += f"\nFormat: '1. description text' (NOT '1-1.' or '1:' or any other format)"
        prompt += "\nDo NOT add preamble, headers, section labels, or markdown formatting."
        prompt += "\nJust the numbered list of improved descriptions, nothing else."
        prompt += f"\n\nExpected output structure:"
        prompt += f"\n1. [improved description for column 1]"
        prompt += f"\n2. [improved description for column 2]"
        if len(columns) > 2:
            prompt += f"\n3. [improved description for column 3]"
            if len(columns) > 3:
                prompt += f"\n..."
                prompt += f"\n{len(columns)}. [improved description for column {len(columns)}]"
        
        return prompt.strip()
    
    def _build_column_batch_prompt(self, context: Dict) -> str:
        """Build batch prompt for multiple columns with sample data awareness"""
        table_name = context['table_name']
        columns = context['columns']
        
        prompt = f"""
Generate professional descriptions for the following columns in table '{table_name}':

"""
        
        for i, col in enumerate(columns, 1):
            sample_values = col.get('sample_values', [])
            
            # Format sample values (with optional redaction)
            should_redact = self.config.get('redact_pii_in_samples', False)
            
            if sample_values:
                samples_preview = []
                for v in sample_values[:3]:  # Show up to 3 samples
                    if v is None:
                        samples_preview.append('NULL')
                    elif should_redact:
                        # Redact all sample values when enabled
                        samples_preview.append('<REDACTED>')
                    else:
                        v_str = str(v)
                        if len(v_str) > 50:
                            v_str = v_str[:47] + '...'
                        samples_preview.append(f"'{v_str}'")
                samples_str = ', '.join(samples_preview)
            else:
                samples_str = '(no samples available)'
            
            prompt += f"""
{i}. Column: {col['column_name']}
   Type: {col['data_type']}
   Sample values: {samples_str}

"""
        
        # Load custom prompt configuration
        prompt_config = self._get_prompt_config()
        
        # Build requirements section based on config
        prompt += "\nFor each column, provide a description that includes:"
        
        if prompt_config.get('include_technical_details', True):
            prompt += "\n- What data it stores (technical details: infer from column name, data type, AND sample values)"
        else:
            prompt += "\n- What data it stores (infer from column name AND sample values)"
        
        if prompt_config.get('include_business_context', True):
            prompt += "\n- Its business purpose and significance in the organization"
        else:
            prompt += "\n- Its purpose in the table"
        
        # Length guidance based on config
        length_guidance = {
            'concise': "Keep descriptions very brief and direct (1 short sentence, ~15-25 words).",
            'standard': "Provide clear, professional descriptions (1-2 sentences, ~30-50 words).",
            'detailed': "Generate comprehensive descriptions with full business context and technical details (2-3 sentences, ~60-100 words)."
        }
        
        # Extract column-specific length (handle both dict and legacy string format)
        length_config = prompt_config.get('description_length', 'standard')
        if isinstance(length_config, dict):
            description_length = length_config.get('column', 'concise')
        else:
            description_length = length_config  # Legacy format
        
        length_instruction = length_guidance.get(description_length, length_guidance['standard'])
        prompt += f"\nLength: {length_instruction}"
        logger.info(f"📏 Applied length setting '{description_length}': {length_instruction}")
        
        # Add custom terminology if defined
        custom_terminology = prompt_config.get('custom_terminology', {})
        if custom_terminology:
            prompt += "\n\nIMPORTANT TERMINOLOGY:"
            for term, meaning in custom_terminology.items():
                prompt += f"\n- '{term}' means '{meaning}'"
            logger.info(f"🔤 Added {len(custom_terminology)} custom terminology mappings to prompt")
        
        # Add additional custom instructions if provided
        additional_instructions = prompt_config.get('additional_instructions', '').strip()
        if additional_instructions:
            prompt += f"\n\nADDITIONAL REQUIREMENTS:\n{additional_instructions}"
            logger.info(f"📝 Added custom instructions to prompt: {additional_instructions[:100]}{'...' if len(additional_instructions) > 100 else ''}")
        
        prompt += f"""

🎯 CRITICAL OUTPUT FORMAT FOR BATCH RESPONSES:
YOU MUST return exactly {len(columns)} descriptions in this EXACT format:

1. Description for first column
2. Description for second column
3. Description for third column
...and so on

STRICT RULES:
- Start each line with the number followed by a period and space: "1. ", "2. ", "3. "
- Do NOT include column names in the descriptions (they're already in the context above)
- Do NOT use bold/italic markers like **text** or *text*
- Do NOT use headers, labels, or prefixes like "Column:", "Description:", "Output:"
- Each description should be plain text starting immediately after the number
- Do NOT add extra blank lines between descriptions
- Do NOT mention specific sample values in descriptions (use them only to infer patterns)

EXAMPLE (for 3 columns):
1. Stores the unique identifier for tracking transactions across systems.
2. Represents the customer's primary email address for communication purposes.
3. Contains the transaction amount in USD for financial reporting and analysis.

Return exactly {len(columns)} numbered descriptions in the format above. This format is MANDATORY for proper parsing."""
        
        return prompt.strip()
    
    def _parse_batch_response(self, response: str, columns: List[Dict]) -> List[str]:
        """Parse LLM batch response into individual descriptions (handles nested/multi-line responses)
        
        Uses aggressive preprocessing to handle format variations from different LLM models.
        """
        if not response or not columns:
            return []
        
        # PREPROCESSING: Normalize various formats (same as table parsing)
        preprocessed_response = response
        preprocessing_applied = []
        
        # Remove bold/italic markers
        if '*' in preprocessed_response or '_' in preprocessed_response:
            preprocessed_response = re.sub(r'[*_]([^*_]+)[*_]', r'\1', preprocessed_response)
            preprocessing_applied.append('markdown_formatting')
        
        # Remove "Column:" or "Description:" labels inline
        if re.search(r'^\d+\.\s+(Column|Description):', preprocessed_response, re.MULTILINE):
            preprocessed_response = re.sub(r'^(\d+\.)\s+(Column|Description):\s*', r'\1 ', preprocessed_response, flags=re.MULTILINE)
            preprocessing_applied.append('inline_labels')
        
        # Handle bullet points → numbers
        lines_with_bullets = [l for l in preprocessed_response.split('\n') if l.strip().startswith('-')]
        if len(lines_with_bullets) >= len(columns):
            line_num = 1
            def replace_bullet(match):
                nonlocal line_num
                result = f"{line_num}. "
                line_num += 1
                return result
            preprocessed_response = re.sub(r'^\s*-\s+', replace_bullet, preprocessed_response, flags=re.MULTILINE)
            preprocessing_applied.append('bullets_to_numbers')
        
        if preprocessing_applied:
            logger.debug(f"🔧 Column parsing preprocessing: {', '.join(preprocessing_applied)}")
        
        lines = preprocessed_response.strip().split('\n')
        descriptions = []
        current_column_text = ""
        in_column_block = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new MAIN column item (e.g., "1. **Column ABC1**", "1. Column: ABC1", "1-1. Column ABC1")
            # Match both "Column:" and "Column " formats, and handle variations like "1." or "1-1."
            if re.match(r'^\d+[-.]?\d*\.\s+(\*\*)?Column[:\s]', line):
                # Save previous column description if exists
                if current_column_text:
                    # Extract only the FIRST sentence/sub-item as the description
                    first_desc = self._extract_first_description(current_column_text)
                    descriptions.append(first_desc)
                
                # Start new column block - extract column name and any following text
                # Handle both "1." and "1-1." formats
                current_column_text = re.sub(r'^\d+[-.]?\d*\.\s+', '', line)
                # Remove markdown bold wrappers (e.g., "**Column ABC1**" -> "Column ABC1")
                current_column_text = re.sub(r'^\*\*Column[:\s]([^*]+)\*\*', r'Column \1', current_column_text)
                # Remove "Column: name" or "Column name" prefix
                current_column_text = re.sub(r'^Column[:\s]+\w+\s*', '', current_column_text)
                in_column_block = True
                logger.debug(f"📝 Found column block, extracted text: {current_column_text[:100]}...")
            elif in_column_block:
                # Accumulate all text under this column (including sub-items)
                current_column_text += " " + line
        
        # Don't forget the last column
        if current_column_text:
            first_desc = self._extract_first_description(current_column_text)
            descriptions.append(first_desc)
        
        # If we didn't find column blocks, try the old simple numbered parsing
        if len(descriptions) == 0:
            logger.debug(f"No column blocks found, using simple numbered parsing")
            current_desc = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Match "1." or "1-1." or similar patterns, but only at top level
                if re.match(r'^\d+[-.]?\d*\.\s+', line) and not re.match(r'^\s+\d+\.', line):  # Top-level numbers only
                    if current_desc:
                        descriptions.append(current_desc.strip())
                    # Strip the number prefix (handles both "1." and "1-1." formats)
                    current_desc = re.sub(r'^\d+[-.]?\d*\.\s+', '', line)
                else:
                    if current_desc:
                        current_desc += " " + line
            if current_desc:
                descriptions.append(current_desc.strip())
        
        # Ensure we have enough descriptions
        if len(descriptions) < len(columns):
            logger.warning(f"⚠️ Parsed only {len(descriptions)} descriptions from LLM, expected {len(columns)}.")
            while len(descriptions) < len(columns):
                descriptions.append(f"Contains {columns[len(descriptions)]['column_name'].replace('_', ' ')} information.")
        
        return descriptions[:len(columns)]
    
    def _extract_first_description(self, text: str) -> str:
        """Extract the first meaningful description from potentially nested text"""
        # Look for the first numbered sub-item (e.g., "1. This column stores..." or "1-1. This column stores...")
        # Handle variations like "1." or "1-1." or "1-2."
        match = re.search(r'\s*\d+[-.]?\d*\.\s+(.+?)(?:\s+\d+[-.]?\d*\.|$)', text, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            # If no sub-items, return the whole text (cleaned)
            result = text.strip()
        
        # Remove common prefixes that LLMs add
        result = re.sub(r'^Description:\s*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'^Column\s+Description:\s*', '', result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def _parse_batch_descriptions(self, response: Any, expected_count: int) -> List[str]:
        """Parse batch LLM response into individual descriptions"""
        import re  # Import at function level
        
        try:
            # Extract text from response
            if isinstance(response, dict):
                if 'choices' in response and len(response['choices']) > 0:
                    response_text = response['choices'][0].get('message', {}).get('content', '')
                elif 'content' in response:
                    response_text = response['content']
                else:
                    response_text = str(response)
            else:
                response_text = str(response)
            
            if not response_text:
                return [f"Description for object {i+1}" for i in range(expected_count)]
            
            # Special case: For single items, just return the full text (no parsing needed)
            if expected_count == 1:
                # Clean up any numbered prefixes that might have been added
                clean_text = response_text.strip()
                # Remove leading "1. " if present
                clean_text = re.sub(r'^1\.\s+', '', clean_text)
                # Remove "Description:" prefix if present
                clean_text = re.sub(r'^Description:\s*', '', clean_text, flags=re.IGNORECASE)
                # Remove bold table names like **table_name**
                clean_text = re.sub(r'^\*\*[^*]+\*\*\s*', '', clean_text)
                return [clean_text.strip()]
            
            # Try multiple parsing strategies
            descriptions = []
            
            # ========================================
            # STAGE 1: AGGRESSIVE PREPROCESSING
            # ========================================
            # Handle multiple format variations that different LLMs produce
            # Goal: Normalize various formats into: "1. Description\n2. Description\n..."
            
            preprocessed_text = response_text
            
            # Track which preprocessing steps were needed (for model behavior analysis)
            preprocessing_applied = []
            
            # Remove bold table name headers: **1. schema.table** → 1.
            if re.search(r'\*\*\d+\.', preprocessed_text):
                preprocessed_text = re.sub(r'\*\*(\d+)\.\s+[^*]+\*\*\s*\n?', r'\1. ', preprocessed_text)
                preprocessing_applied.append('bold_headers')
            
            # Remove italic markers: *text* or _text_ → text
            if '*' in preprocessed_text or '_' in preprocessed_text:
                preprocessed_text = re.sub(r'[*_]([^*_]+)[*_]', r'\1', preprocessed_text)
                preprocessing_applied.append('italic_markers')
            
            # Remove markdown headers: ### Header or ## Header → Header
            if '#' in preprocessed_text:
                preprocessed_text = re.sub(r'^#+\s*', '', preprocessed_text, flags=re.MULTILINE)
                preprocessing_applied.append('markdown_headers')
            
            # Handle hyphenated list format: "- Description" → "1. Description"
            # (Some models use bullet points instead of numbers)
            lines_with_hyphens = [l for l in preprocessed_text.split('\n') if l.strip().startswith('-')]
            if len(lines_with_hyphens) >= expected_count:
                line_num = 1
                def replace_hyphen(match):
                    nonlocal line_num
                    result = f"{line_num}. "
                    line_num += 1
                    return result
                preprocessed_text = re.sub(r'^\s*-\s+', replace_hyphen, preprocessed_text, flags=re.MULTILINE)
                preprocessing_applied.append('hyphen_to_number')
            
            # Handle "Table 1:", "Item 1:", "Description 1:" prefixes → "1. "
            if re.search(r'(Table|Item|Description)\s+\d+:', preprocessed_text, re.IGNORECASE):
                preprocessed_text = re.sub(r'(?:Table|Item|Description)\s+(\d+):\s*', r'\1. ', preprocessed_text, flags=re.IGNORECASE)
                preprocessing_applied.append('prefix_labels')
            
            # Remove "Description:" or "Output:" labels
            if 'Description:' in preprocessed_text or 'Output:' in preprocessed_text:
                preprocessed_text = re.sub(r'^(Description|Output):\s*', '', preprocessed_text, flags=re.IGNORECASE | re.MULTILINE)
                preprocessing_applied.append('description_labels')
            
            if preprocessing_applied:
                logger.info(f"🔧 Applied preprocessing steps: {', '.join(preprocessing_applied)}")
            
            # Strategy 1: Try numbered format (1., 2., 3., etc.)
            lines = preprocessed_text.strip().split('\n')
            current_desc = ""
            found_first_item = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a numbered line
                if re.match(r'^\d+\.\s+', line):
                    # Save previous description if exists
                    if current_desc:
                        descriptions.append(current_desc.strip())
                    # Start new description (remove number)
                    current_desc = re.sub(r'^\d+\.\s+', '', line)
                    found_first_item = True
                else:
                    # Continue current description (but ONLY if we've found the first numbered item)
                    if found_first_item and current_desc:
                        current_desc += " " + line
            
            # Don't forget the last description
            if current_desc:
                descriptions.append(current_desc.strip())
            
            # Strategy 2: If numbered format didn't work, try newline-separated paragraphs
            if len(descriptions) < expected_count:
                logger.debug(f"Numbered format found {len(descriptions)} items, trying paragraph format")
                descriptions = []
                
                # Use preprocessed text (already has bold markers removed)
                # Also remove any remaining bold markers in the description text itself
                cleaned_text = re.sub(r'\*\*[^*]+\*\*\s*\n?', '', preprocessed_text)
                
                # Split by double newlines OR single newlines if we have enough sentences
                # First try double newlines (paragraph breaks)
                paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
                
                if len(paragraphs) >= expected_count:
                    # We have enough paragraphs
                    descriptions = paragraphs
                else:
                    # Try single newlines - each non-empty line is a description
                    single_lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
                    
                    # Filter out very short lines (likely headers or noise)
                    meaningful_lines = [line for line in single_lines if len(line) > 30]
                    
                    if len(meaningful_lines) >= expected_count:
                        descriptions = meaningful_lines
                    elif len(single_lines) >= expected_count:
                        # Use all lines if meaningful filtering removed too many
                        descriptions = single_lines
            
            # Strategy 3: If still not enough, try splitting by sentence patterns
            # Some models return descriptions in a single block separated by periods
            if len(descriptions) < expected_count:
                logger.debug(f"Paragraph format found {len(descriptions)} items, trying sentence splitting")
                # Split on patterns like ". \n" or ". [Capital letter]"
                sentences = re.split(r'\.\s+(?=[A-Z])', preprocessed_text.strip())
                sentences = [s.strip() + ('.' if not s.endswith('.') else '') for s in sentences if s.strip()]
                
                if len(sentences) >= expected_count:
                    descriptions = sentences
            
            # ========================================
            # STAGE 2: INTELLIGENT FALLBACK
            # ========================================
            # If all parsing strategies failed, try to extract ANY useful content
            if len(descriptions) < expected_count:
                logger.warning(f"⚠️ Standard parsing yielded {len(descriptions)}/{expected_count} descriptions, trying intelligent fallback")
                
                # Split text into any substantial chunks (30+ chars) and take the first N
                all_text_chunks = preprocessed_text.split('\n')
                substantial_chunks = [chunk.strip() for chunk in all_text_chunks 
                                     if len(chunk.strip()) > 30 and not chunk.strip().startswith('#')]
                
                if len(substantial_chunks) >= expected_count:
                    logger.info(f"✅ Fallback extracted {len(substantial_chunks)} substantial text chunks")
                    descriptions = substantial_chunks[:expected_count]
                else:
                    # Last resort: Split by periods and take sentences
                    sentences = [s.strip() + '.' for s in preprocessed_text.split('.') if len(s.strip()) > 20]
                    if len(sentences) >= expected_count:
                        logger.info(f"✅ Fallback extracted {len(sentences)} sentences")
                        descriptions = sentences[:expected_count]
            
            # ========================================
            # STAGE 3: CLEANUP & NORMALIZATION
            # ========================================
            cleaned_descriptions = []
            for desc in descriptions:
                # Remove any remaining markdown bold markers
                desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', desc)
                # Remove "Description:" prefix
                desc = re.sub(r'^Description:\s*', '', desc, flags=re.IGNORECASE)
                # Remove leading numbers if any slipped through
                desc = re.sub(r'^\d+\.\s+', '', desc)
                # Remove table names if present (e.g., "schema.table - " or "schema.table: ")
                desc = re.sub(r'^[\w\d_]+\.[\w\d_]+[\s\-:]+', '', desc)
                cleaned_descriptions.append(desc.strip())
            
            descriptions = cleaned_descriptions
            
            # ========================================
            # STAGE 4: FINAL SAFETY NET
            # ========================================
            # Only use generic fallbacks as absolute last resort
            while len(descriptions) < expected_count:
                logger.error(f"❌ PARSING FAILED: Only {len(descriptions)}/{expected_count} descriptions extracted")
                logger.error(f"📊 Model used: {self.config.get('model', 'unknown')}")
                logger.error(f"📝 Response preview: {response_text[:500]}...")
                descriptions.append(f"Description for object {len(descriptions) + 1}")
            
            # Truncate if we have too many
            return descriptions[:expected_count]
            
        except Exception as e:
            logger.error(f"Failed to parse batch descriptions: {e}")
            return [f"Description for object {i+1}" for i in range(expected_count)]
    
    def _calculate_confidence_score(self, description: str, context: Dict, object_type: str, pii_analysis: Dict = None) -> float:
        """
        Calculate confidence score for generated description using fast heuristics.
        No LLM calls - optimized for batch processing hundreds of objects.
        
        Score components:
        - Lexical diversity (30%): Unique words / total words
        - Context utilization (25%): References to actual context data
        - Specificity (25%): Domain terms vs generic terms
        - Structure quality (20%): Well-formed, appropriate length
        """
        if not description or description.strip() == "":
            return 0.0
        
        desc_lower = description.lower()
        words = desc_lower.split()
        
        # Component 1: Lexical Diversity (30%) - Fast unique word ratio
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            lexical_score = min(unique_ratio * 1.2, 1.0)  # Boost slightly, cap at 1.0
        else:
            lexical_score = 0.0
        
        # Component 2: Context Utilization (25%) - Did LLM use provided context?
        context_score = 0.0
        if object_type == 'schema' and context.get('table_names'):
            # Check if description mentions any actual table names
            table_names = [t.lower() for t in context.get('table_names', [])]
            mentioned = sum(1 for t in table_names if t in desc_lower)
            context_score = min(mentioned / max(len(table_names), 1) * 2, 1.0)
        elif object_type == 'table' and context.get('column_names'):
            # Check if description mentions column names or data types
            col_names = [c.lower() for c in context.get('column_names', [])]
            mentioned = sum(1 for c in col_names[:10] if c in desc_lower)  # Check first 10
            context_score = min(mentioned / min(len(col_names), 5), 1.0)
        elif object_type == 'column' and context.get('sample_values'):
            # For columns, just check if it has samples (already used)
            context_score = 0.8
        
        # Component 3: Specificity (25%) - Domain-specific vs generic terms
        domain_terms = [
            'join', 'foreign key', 'primary key', 'index', 'aggregate', 'metric',
            'dimension', 'fact', 'relationship', 'reference', 'lookup', 'transaction',
            'temporal', 'snapshot', 'incremental', 'partition', 'denormalized'
        ]
        generic_terms = [
            'contains', 'stores', 'holds', 'has', 'includes', 'information', 
            'data', 'values', 'records', 'items'
        ]
        
        domain_count = sum(1 for term in domain_terms if term in desc_lower)
        generic_count = sum(1 for term in generic_terms if term in desc_lower)
        
        # Reward domain terms, penalize excessive generic terms
        specificity_score = min((domain_count * 0.3) + max(0, (1 - generic_count * 0.2)), 1.0)
        
        # Component 4: Structure Quality (20%) - Length and sentence structure
        char_count = len(description)
        sentence_count = description.count('.') + description.count('!') + description.count('?')
        
        # Optimal length ranges (based on generation style)
        if object_type == 'schema':
            optimal_min, optimal_max = 100, 300
        elif object_type == 'table':
            optimal_min, optimal_max = 80, 250
        else:  # column
            optimal_min, optimal_max = 40, 150
        
        # Length score: Peak at optimal range, drop off outside
        if char_count < optimal_min:
            length_score = char_count / optimal_min
        elif char_count <= optimal_max:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - ((char_count - optimal_max) / optimal_max))
        
        # Sentence count score: 1-3 sentences is ideal
        if sentence_count >= 1 and sentence_count <= 3:
            sentence_score = 1.0
        elif sentence_count == 0:
            sentence_score = 0.3
        else:
            sentence_score = max(0.5, 1.0 - (sentence_count - 3) * 0.1)
        
        structure_score = (length_score * 0.7) + (sentence_score * 0.3)
        
        # Weighted combination
        final_score = (
            lexical_score * 0.30 +
            context_score * 0.25 +
            specificity_score * 0.25 +
            structure_score * 0.20
        )
        
        # Bonus: PII detected (shows thorough analysis)
        if pii_analysis and pii_analysis.get('pii_types'):
            final_score = min(final_score + 0.05, 1.0)
        
        return round(final_score, 2)
    
    def _create_optimal_chunks(self, columns: List[Dict]) -> List[List[Dict]]:
        """Create optimal chunks for column processing"""
        if not columns:
            return []
        
        chunk_size = self.config['max_chunk_size']
        chunks = []
        
        for i in range(0, len(columns), chunk_size):
            chunk = columns[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    async def _get_tables_in_schema(self, catalog_name: str, schema_name: str) -> List[Dict]:
        """Get tables in a schema for context"""
        try:
            # Use the Unity service to get tables in the schema
            tables = self.unity_service.get_tables_with_missing_metadata(catalog_name)
            # Filter to only tables in this specific schema
            schema_tables = [t for t in tables if t.get('schema_name') == schema_name]
            
            # Also get tables that already have metadata (not just missing ones)
            try:
                # Call Unity Catalog API directly for all tables in schema
                token = self.unity_service._get_oauth_token()
                headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
                url = f"https://{self.unity_service.workspace_host}/api/2.1/unity-catalog/tables"
                params = {'catalog_name': catalog_name, 'schema_name': schema_name}
                
                import requests
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                all_tables = response.json().get('tables', [])
                
                logger.info(f"Found {len(all_tables)} tables in schema {catalog_name}.{schema_name}")
                return all_tables
                
            except Exception as api_error:
                logger.warning(f"Could not get all tables via API for {catalog_name}.{schema_name}: {api_error}")
                # Fallback to just the missing metadata tables
                return schema_tables
                
        except Exception as e:
            logger.error(f"Error getting tables for schema {catalog_name}.{schema_name}: {e}")
            return []
    
    async def _get_table_columns_metadata(self, catalog_name: str, schema_name: str, table_name: str) -> List[Dict]:
        """
        Get table column metadata (names, types) WITHOUT sampling data.
        
        Used for: Table-level description generation (fast, no data access needed)
        Limits to max_table_context_columns to prevent wide table slowdown.
        
        CACHED: Results are cached across models to avoid redundant queries.
        """
        try:
            full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
            cache_key = full_table_name
            
            # Check cache first
            if cache_key in self.column_metadata_cache:
                logger.debug(f"♻️ Column metadata cache hit for {full_table_name}")
                return self.column_metadata_cache[cache_key]
            
            # Cap columns at configured limit for table context (prevents wide table slowdown)
            max_cols = self.config.get('max_table_context_columns', 50)
            
            # Get column metadata from information_schema (no data sampling)
            columns_sql = f"""
                SELECT column_name, data_type, comment
                FROM system.information_schema.columns
                WHERE table_catalog = '{catalog_name}'
                    AND table_schema = '{schema_name}'
                    AND table_name = '{table_name}'
                ORDER BY ordinal_position
                LIMIT {max_cols}
            """
            
            columns_metadata = self.unity_service._execute_sql_warehouse(columns_sql)
            
            if not columns_metadata:
                logger.warning(f"No columns found for {full_table_name}")
                return []
            
            # Return column metadata without sample values (lightweight)
            columns = [
                {
                    'name': row[0],
                    'data_type': row[1],
                    'comment': row[2] if len(row) > 2 else None,
                    'sample_values': []  # No sampling for table descriptions
                }
                for row in columns_metadata
            ]
            
            # Cache the result
            self.column_metadata_cache[cache_key] = columns
            
            if len(columns) >= max_cols:
                logger.info(f"✅ Retrieved {len(columns)} column names for {full_table_name} (capped at {max_cols} for table context, no data sampling)")
            else:
                logger.info(f"✅ Retrieved {len(columns)} column names for {full_table_name} (no data sampling)")
            return columns
            
        except Exception as e:
            logger.error(f"❌ Error getting column metadata for {catalog_name}.{schema_name}.{table_name}: {e}")
            return []
    
    async def _get_table_columns_with_samples(self, catalog_name: str, schema_name: str, table_name: str) -> List[Dict]:
        """
        Get table columns WITH actual data samples from the table.
        
        Used for: Column-level description generation (requires actual data)
        Only runs when generating column descriptions, not table descriptions.
        
        CACHED: Sample data is cached across models to avoid redundant table scans.
        """
        try:
            full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
            cache_key = full_table_name
            
            # Check cache first
            if cache_key in self.sample_data_cache:
                logger.debug(f"♻️ Sample data cache hit for {full_table_name}")
                return self.sample_data_cache[cache_key]
            
            # Check if sampling is enabled
            sampling_enabled = self.config.get('enable_sampling', True)
            if not sampling_enabled:
                logger.info(f"📊 Sampling disabled (enable_sampling={sampling_enabled}) for {full_table_name}, returning metadata only")
                # Fall back to metadata-only (no samples)
                return await self._get_table_columns_metadata(catalog_name, schema_name, table_name)
            
            sample_rows = self.config.get('sample_rows', 10)
            # Cap columns at configured limit to prevent wide table slowdown
            max_cols = self.config.get('max_table_context_columns', 50)
            logger.info(f"📊 Sampling {sample_rows} rows from {full_table_name} for column analysis (max {max_cols} columns)")
            
            # Get column metadata from information_schema (capped for wide tables)
            columns_sql = f"""
                SELECT column_name, data_type, comment
                FROM system.information_schema.columns
                WHERE table_catalog = '{catalog_name}'
                    AND table_schema = '{schema_name}'
                    AND table_name = '{table_name}'
                ORDER BY ordinal_position
                LIMIT {max_cols}
            """
            
            columns_metadata = self.unity_service._execute_sql_warehouse(columns_sql)
            
            if not columns_metadata:
                logger.warning(f"No columns found for {full_table_name}")
                return []
            
            # Sample actual data from the table (ONLY for column-level generation)
            # Build explicit column list to guarantee correct order
            column_names_ordered = [row[0] for row in columns_metadata]
            columns_list = ', '.join([f"`{col}`" for col in column_names_ordered])
            sample_sql = f"SELECT {columns_list} FROM {full_table_name} LIMIT {sample_rows}"
            
            try:
                sample_data = self.unity_service._execute_sql_warehouse(sample_sql)
            except Exception as e:
                logger.warning(f"Could not sample data from {full_table_name}: {e}")
                # Return columns without samples if sampling fails
                return [
                    {
                        'name': row[0],
                        'data_type': row[1],
                        'comment': row[2] if len(row) > 2 else None,
                        'sample_values': []
                    }
                    for row in columns_metadata
                ]
            
            # Build result with columns and their sample values
            # Now col_index is guaranteed to match the SELECT order
            columns_with_samples = []
            for col_index, col_metadata in enumerate(columns_metadata):
                column_name = col_metadata[0]
                data_type = col_metadata[1]
                comment = col_metadata[2] if len(col_metadata) > 2 else None
                
                # Extract sample values for this column from all sampled rows
                sample_values = []
                for row in sample_data:
                    if col_index < len(row) and row[col_index] is not None:
                        sample_values.append(row[col_index])
                
                columns_with_samples.append({
                    'name': column_name,
                    'data_type': data_type,
                    'comment': comment,
                    'sample_values': sample_values[:10]  # Limit to 10 samples for analysis
                })
            
            # Cache the sample data
            self.sample_data_cache[cache_key] = columns_with_samples
            
            if len(columns_with_samples) >= max_cols:
                logger.info(f"✅ Sampled {len(columns_with_samples)} columns with data from {full_table_name} (capped at {max_cols} to prevent wide table slowdown)")
            else:
                logger.info(f"✅ Sampled {len(columns_with_samples)} columns with data from {full_table_name}")
            return columns_with_samples
            
        except Exception as e:
            logger.error(f"❌ Error sampling data from {catalog_name}.{schema_name}.{table_name}: {e}")
            return []
    
    async def _enrich_columns_with_samples(self, columns: List[Dict], table_sample_cache: Dict = None) -> List[Dict]:
        """Enrich column information with sample data for PII detection and context
        
        NOTE: This is now called per-chunk (typically 10-20 columns) rather than per-table,
        which dramatically improves performance for wide tables (100+ columns).
        
        Args:
            columns: List of column dicts to enrich
            table_sample_cache: Optional pre-cached sample data for the table to avoid re-querying
        """
        if not columns:
            return columns
        
        # Check if sampling is enabled
        sampling_enabled = self.config.get('enable_sampling', True)
        if not sampling_enabled:
            logger.info(f"📊 Sampling disabled (enable_sampling={sampling_enabled}), returning columns without samples")
            return columns
        
        # All columns should be from the same table
        first_col = columns[0]
        catalog_name = first_col['catalog_name']
        schema_name = first_col['schema_name']
        table_name = first_col['table_name']
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
        
        # OPTIMIZATION: Check if we have cached sample data for this table
        if table_sample_cache is not None:
            logger.info(f"📦 Using cached sample data for {full_table_name} (avoiding redundant query)")
            # Enrich columns with cached sample data
            for col in columns:
                col_name = col['column_name']
                if col_name in table_sample_cache:
                    col['sample_values'] = table_sample_cache[col_name].get('sample_values', [])
            return columns
        
        sample_rows = self.config.get('sample_rows', 10)
        max_cols = self.config.get('max_table_context_columns', 50)
        
        # OPTIMIZATION: Only sample the columns in this chunk, not the entire table
        # Extract the column names we actually need to sample
        columns_to_sample = [col['column_name'] for col in columns]
        
        # Cap at max_table_context_columns to prevent edge cases with huge chunks
        # NOTE: With per-chunk sampling, chunks are typically 10-20 columns, so this rarely triggers
        # Columns beyond max_cols will be processed without sample data
        if len(columns_to_sample) > max_cols:
            logger.warning(f"⚠️ Large chunk: {len(columns_to_sample)} columns, sampling first {max_cols} for performance (remaining {len(columns_to_sample) - max_cols} in this chunk will process without samples)")
            columns_to_sample = columns_to_sample[:max_cols]
        
        # Typically chunks are small now, so this is very efficient
        logger.info(f"📊 Sampling {sample_rows} rows from {full_table_name} for {len(columns_to_sample)} columns in this chunk")
        
        try:
            # Get column names in the correct order from information_schema
            # Only fetch the columns we need, in their ordinal position order
            columns_sql = f"""
                SELECT column_name
                FROM system.information_schema.columns
                WHERE table_catalog = '{catalog_name}'
                    AND table_schema = '{schema_name}'
                    AND table_name = '{table_name}'
                    AND column_name IN ({','.join([f"'{col}'" for col in columns_to_sample])})
                ORDER BY ordinal_position
            """
            
            try:
                columns_metadata = self.unity_service._execute_sql_warehouse(columns_sql)
                ordered_column_names = [row[0] for row in columns_metadata]
            except Exception as e:
                logger.warning(f"Could not get column names from {full_table_name}: {e}")
                return columns
            
            if not ordered_column_names:
                logger.warning(f"No columns found in information_schema for {full_table_name}")
                return columns
            
            # Build SELECT query ONLY for the columns we need (much more efficient for wide tables)
            columns_list = ', '.join([f"`{col}`" for col in ordered_column_names])
            sample_sql = f"SELECT {columns_list} FROM {full_table_name} LIMIT {sample_rows}"
            
            logger.info(f"📊 Optimized sample query: SELECT {len(ordered_column_names)} columns FROM {full_table_name} LIMIT {sample_rows}")
            
            try:
                sample_data = self.unity_service._execute_sql_warehouse(sample_sql)
            except Exception as e:
                logger.warning(f"Could not sample data from {full_table_name}: {e}")
                # Return columns without samples if sampling fails
                return columns
            
            if not sample_data:
                logger.info(f"No sample data returned from {full_table_name}")
                return columns
            
            # Build column name to index mapping (now guaranteed to match SELECT order)
            column_positions = {col_name: idx for idx, col_name in enumerate(ordered_column_names)}
            
            # Enrich each column with its sample values
            enriched_columns = []
            for column in columns:
                column_name = column['column_name']
                col_index = column_positions.get(column_name)
                
                if col_index is None:
                    # Column wasn't sampled (likely beyond max_cols limit or missing)
                    enriched_columns.append(column)
                    continue
                
                # Extract sample values for this column from all sampled rows
                sample_values = []
                for row in sample_data:
                    if col_index < len(row) and row[col_index] is not None:
                        sample_values.append(row[col_index])
                
                # Add sample values to column dict
                enriched_column = column.copy()
                enriched_column['sample_values'] = sample_values[:10]  # Limit to 10 samples
                enriched_columns.append(enriched_column)
            
            # Count how many columns actually got samples
            cols_with_data = sum(1 for col in enriched_columns if col.get('sample_values'))
            logger.info(f"✅ Enriched {len(enriched_columns)} columns: {cols_with_data} have sample data, {len(enriched_columns) - cols_with_data} are empty/NULL")
            return enriched_columns
            
        except Exception as e:
            logger.error(f"❌ Error enriching columns with samples from {full_table_name}: {e}")
            return columns  # Return original columns on error
    
    def _prepare_column_samples_for_context(self, columns_info: List[Dict], pii_analysis: Optional[Dict], redact_pii: bool = False) -> List[Dict]:
        """
        Prepare column sample values for inclusion in LLM context.
        
        Args:
            columns_info: List of column dictionaries with 'name', 'data_type', and 'sample_values'
            pii_analysis: PII analysis results containing column-level PII detection
            redact_pii: If True, replace PII values with <REDACTED>
        
        Returns:
            List of dicts with column name, type, and sample values (optionally redacted)
        """
        prepared_samples = []
        
        # Build a set of PII column names for quick lookup
        pii_columns = set()
        if redact_pii and pii_analysis:
            for col_analysis in pii_analysis.get('column_analysis', []):
                if col_analysis.get('pii_types'):
                    pii_columns.add(col_analysis['column_name'])
        
        for col in columns_info:
            col_name = col.get('name', '')
            data_type = col.get('data_type', '')
            sample_values = col.get('sample_values', [])
            
            if not sample_values:
                continue
            
            # Take first 3-5 samples (depending on how many we have)
            limited_samples = sample_values[:5]
            
            # Process each sample value
            processed_samples = []
            for sample in limited_samples:
                if sample is None:
                    processed_samples.append('NULL')
                    continue
                
                # Convert to string and truncate if too long
                sample_str = str(sample)
                if len(sample_str) > 50:
                    sample_str = sample_str[:47] + '...'
                
                # Redact if this column has PII and redaction is enabled
                if redact_pii and col_name in pii_columns:
                    processed_samples.append('<REDACTED>')
                else:
                    processed_samples.append(sample_str)
            
            prepared_samples.append({
                'column_name': col_name,
                'data_type': data_type,
                'sample_values': processed_samples
            })
        
        return prepared_samples
    
    def _get_selected_schemas_metadata(self, catalog_name: str, selected_schema_names: List[str]) -> List[Dict]:
        """Get metadata for only the selected schemas (optimized)"""
        if not selected_schema_names:
            return []
        
        try:
            # Build SQL query for only the selected schemas
            schema_list = "', '".join(selected_schema_names)
            sql_query = f"""
                SELECT schema_name, schema_owner, created, last_altered
                FROM system.information_schema.schemata
                WHERE catalog_name = '{catalog_name}' 
                AND schema_name IN ('{schema_list}')
                AND (comment IS NULL OR comment = '' OR LENGTH(TRIM(comment)) = 0)
            """
            
            data = self.unity_service._execute_sql_warehouse(sql_query)
            missing_metadata = []
            
            for row in data:
                schema_name = row[0] if len(row) > 0 else ''
                schema_owner = row[1] if len(row) > 1 else ''
                created = row[2] if len(row) > 2 else None
                updated = row[3] if len(row) > 3 else None
                
                missing_metadata.append({
                    'name': schema_name,
                    'full_name': f"{catalog_name}.{schema_name}",
                    'catalog_name': catalog_name,
                    'comment': '',
                    'owner': schema_owner,
                    'created_at': created,
                    'updated_at': updated
                })
            
            logger.info(f"🚀 Found {len(missing_metadata)} selected schemas with missing descriptions using optimized SQL")
            return missing_metadata
            
        except Exception as e:
            logger.warning(f"Optimized schema query failed, using fallback: {e}")
            # Fallback to full scan + filter
            all_schemas = self.unity_service.get_schemas_with_missing_metadata(catalog_name)
            return [s for s in all_schemas if s['name'] in selected_schema_names]
    
    def _get_selected_tables_metadata(self, catalog_name: str, selected_table_names: List[str]) -> List[Dict]:
        """Get metadata for only the selected tables (optimized)"""
        if not selected_table_names:
            return []
        
        try:
            # Build SQL query for only the selected tables
            table_conditions = []
            for full_name in selected_table_names:
                parts = full_name.split('.')
                if len(parts) >= 3:
                    catalog, schema, table = parts[0], parts[1], parts[2]
                    table_conditions.append(f"(table_catalog = '{catalog}' AND table_schema = '{schema}' AND table_name = '{table}')")
            
            if not table_conditions:
                return []
            
            conditions_sql = " OR ".join(table_conditions)
            sql_query = f"""
                SELECT table_catalog, table_schema, table_name, table_type, table_owner, created, last_altered
                FROM system.information_schema.tables
                WHERE ({conditions_sql})
                AND (comment IS NULL OR comment = '' OR LENGTH(TRIM(comment)) = 0)
            """
            
            data = self.unity_service._execute_sql_warehouse(sql_query)
            missing_metadata = []
            
            for row in data:
                catalog = row[0] if len(row) > 0 else ''
                schema_name = row[1] if len(row) > 1 else ''
                table_name = row[2] if len(row) > 2 else ''
                table_type = row[3] if len(row) > 3 else ''
                owner = row[4] if len(row) > 4 else ''
                created = row[5] if len(row) > 5 else None
                updated = row[6] if len(row) > 6 else None
                
                missing_metadata.append({
                    'name': table_name,
                    'full_name': f"{catalog}.{schema_name}.{table_name}",
                    'catalog_name': catalog,
                    'schema_name': schema_name,
                    'table_name': table_name,
                    'table_type': table_type,
                    'comment': '',
                    'owner': owner,
                    'created_at': created,
                    'updated_at': updated
                })
            
            logger.info(f"🚀 Found {len(missing_metadata)} selected tables with missing descriptions using optimized SQL")
            return missing_metadata
            
        except Exception as e:
            logger.warning(f"Optimized table query failed, using fallback: {e}")
            # Fallback to full scan + filter
            all_tables = self.unity_service.get_tables_with_missing_metadata(catalog_name)
            return [t for t in all_tables if t.get('full_name', '') in selected_table_names]
    
    def _get_selected_columns_metadata(self, catalog_name: str, selected_column_names: List[str]) -> List[Dict]:
        """Get metadata for only the selected columns (optimized)"""
        if not selected_column_names:
            return []
        
        try:
            # Build SQL query for only the selected columns
            column_conditions = []
            for full_name in selected_column_names:
                parts = full_name.split('.')
                if len(parts) >= 4:
                    catalog, schema, table, column = parts[0], parts[1], parts[2], parts[3]
                    column_conditions.append(f"(table_catalog = '{catalog}' AND table_schema = '{schema}' AND table_name = '{table}' AND column_name = '{column}')")
            
            if not column_conditions:
                return []
            
            conditions_sql = " OR ".join(column_conditions)
            sql_query = f"""
                SELECT table_catalog, table_schema, table_name, column_name, data_type, is_nullable, column_default
                FROM system.information_schema.columns
                WHERE ({conditions_sql})
                AND (comment IS NULL OR comment = '' OR LENGTH(TRIM(comment)) = 0)
            """
            
            data = self.unity_service._execute_sql_warehouse(sql_query)
            missing_metadata = []
            
            for row in data:
                catalog = row[0] if len(row) > 0 else ''
                schema_name = row[1] if len(row) > 1 else ''
                table_name = row[2] if len(row) > 2 else ''
                column_name = row[3] if len(row) > 3 else ''
                data_type = row[4] if len(row) > 4 else ''
                is_nullable = row[5] if len(row) > 5 else ''
                column_default = row[6] if len(row) > 6 else ''
                
                missing_metadata.append({
                    'name': column_name,
                    'full_name': f"{catalog}.{schema_name}.{table_name}.{column_name}",
                    'catalog_name': catalog,
                    'schema_name': schema_name,
                    'table_name': table_name,
                    'column_name': column_name,
                    'data_type': data_type,
                    'is_nullable': is_nullable,
                    'column_default': column_default,
                    'comment': ''
                })
            
            logger.info(f"🚀 Found {len(missing_metadata)} selected columns with missing comments using optimized SQL")
            return missing_metadata
            
        except Exception as e:
            logger.warning(f"Optimized column query failed, using fallback: {e}")
            # Fallback to full scan + filter
            all_columns = self.unity_service.get_columns_with_missing_metadata(catalog_name)
            return [c for c in all_columns if c.get('full_name', '') in selected_column_names]
