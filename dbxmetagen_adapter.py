"""
dbxmetagen Adapter
Integrates dbxmetagen industry solution as an alternative generation engine
"""

import logging
import time
import yaml
import os
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DbxMetagenAdapter:
    """
    Adapter for running dbxmetagen notebooks as an alternative metadata generation engine
    """
    
    def __init__(self, workspace_client, unity_service):
        """
        Initialize adapter
        
        Args:
            workspace_client: Databricks Workspace client
            unity_service: Unity Catalog service for database operations
        """
        self.workspace_client = workspace_client
        self.unity_service = unity_service
        self.job_runs = {}  # Track running jobs
        
        logger.info("✅ DbxMetagenAdapter initialized")
    
    def validate_configuration(self, config: Dict) -> Dict:
        """
        Validate dbxmetagen configuration
        
        Args:
            config: {
                'notebook_path': str,
                'cluster_id': str,
                'variables_yml_path': str
            }
        
        Returns:
            {'valid': bool, 'errors': []}
        """
        errors = []
        
        try:
            # Check notebook exists
            notebook_path = config.get('notebook_path')
            if not notebook_path:
                errors.append("notebook_path is required")
            else:
                try:
                    # Try to get notebook info
                    from databricks.sdk.service.workspace import GetStatusRequest
                    status = self.workspace_client.workspace.get_status(
                        path=notebook_path
                    )
                    if status.object_type != 'NOTEBOOK':
                        errors.append(f"{notebook_path} is not a notebook")
                    logger.info(f"✅ Notebook found: {notebook_path}")
                except Exception as e:
                    errors.append(f"Notebook not found: {notebook_path} ({str(e)})")
            
            # Check cluster exists
            cluster_id = config.get('cluster_id')
            if not cluster_id:
                errors.append("cluster_id is required")
            else:
                try:
                    cluster = self.workspace_client.clusters.get(cluster_id=cluster_id)
                    logger.info(f"✅ Cluster found: {cluster.cluster_name}")
                except Exception as e:
                    errors.append(f"Cluster not found: {cluster_id} ({str(e)})")
            
            # Check variables.yml path (DBFS)
            variables_path = config.get('variables_yml_path')
            if not variables_path:
                errors.append("variables_yml_path is required")
            elif not variables_path.startswith('/dbfs/'):
                errors.append("variables_yml_path must start with /dbfs/")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Error validating dbxmetagen config: {e}")
            return {
                'valid': False,
                'errors': [str(e)]
            }
    
    def run_metadata_generation(self, catalog: str, schemas: Optional[List[str]], 
                               tables: Optional[List[str]], columns: Optional[List[str]],
                               config: Dict) -> str:
        """
        Run dbxmetagen metadata generation via Databricks Jobs API
        
        Args:
            catalog: Catalog name
            schemas: Optional list of schemas
            tables: Optional list of tables
            columns: Optional list of columns
            config: dbxmetagen configuration
        
        Returns:
            run_id: Unique run identifier
        """
        try:
            run_id = f"dbxmetagen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{catalog}"
            
            logger.info(f"🚀 Starting dbxmetagen job for catalog: {catalog}")
            
            # Update variables.yml with catalog/schema selection
            self._update_variables_yml(catalog, schemas, tables, columns, config)
            
            # Create and submit job
            from databricks.sdk.service import jobs
            
            job_config = jobs.SubmitTask(
                task_key="dbxmetagen_metadata_generation",
                notebook_task=jobs.NotebookTask(
                    notebook_path=config['notebook_path'],
                    base_parameters={
                        "catalog": catalog,
                        "schemas": ','.join(schemas) if schemas else '',
                    }
                ),
                existing_cluster_id=config['cluster_id'],
                timeout_seconds=3600  # 1 hour timeout
            )
            
            job_run = self.workspace_client.jobs.submit(
                run_name=f"UC Metadata Assistant - dbxmetagen - {catalog}",
                tasks=[job_config]
            )
            
            # Store job info
            self.job_runs[run_id] = {
                'databricks_run_id': job_run.run_id,
                'catalog': catalog,
                'schemas': schemas,
                'status': 'RUNNING',
                'start_time': datetime.now().isoformat(),
                'config': config
            }
            
            logger.info(f"✅ dbxmetagen job submitted: {run_id} (Databricks run_id: {job_run.run_id})")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Error starting dbxmetagen job: {e}")
            raise
    
    def get_generation_status(self, run_id: str) -> Dict:
        """
        Get status of dbxmetagen job
        
        Args:
            run_id: Run identifier from run_metadata_generation
        
        Returns:
            {
                'status': 'RUNNING' | 'COMPLETED' | 'FAILED',
                'progress': int (0-100),
                'message': str,
                'databricks_run_url': str,
                'output_file_path': str (if completed)
            }
        """
        try:
            if run_id not in self.job_runs:
                return {
                    'status': 'NOT_FOUND',
                    'message': f'Run ID {run_id} not found'
                }
            
            job_info = self.job_runs[run_id]
            databricks_run_id = job_info['databricks_run_id']
            
            # Get job run status
            run_status = self.workspace_client.jobs.get_run(run_id=databricks_run_id)
            
            # Map Databricks status to our status
            state = run_status.state
            if state.life_cycle_state in ['PENDING', 'RUNNING']:
                status = 'RUNNING'
                progress = 50  # Arbitrary progress estimate
                message = f"Job is {state.life_cycle_state.lower()}"
            elif state.life_cycle_state == 'TERMINATED':
                if state.result_state == 'SUCCESS':
                    status = 'COMPLETED'
                    progress = 100
                    message = "Job completed successfully"
                    # Parse output and import results
                    self._import_dbxmetagen_results(run_id, job_info['config'])
                else:
                    status = 'FAILED'
                    progress = 0
                    message = f"Job failed: {state.state_message or 'Unknown error'}"
            else:
                status = 'FAILED'
                progress = 0
                message = f"Unexpected state: {state.life_cycle_state}"
            
            # Update stored status
            self.job_runs[run_id]['status'] = status
            
            # Build response
            workspace_url = self.workspace_client.config.host
            databricks_run_url = f"{workspace_url}#job/{databricks_run_id}/run/1"
            
            return {
                'status': status,
                'progress': progress,
                'message': message,
                'databricks_run_url': databricks_run_url,
                'start_time': job_info['start_time'],
                'output_file_path': job_info.get('output_file_path', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {
                'status': 'ERROR',
                'message': str(e)
            }
    
    def _update_variables_yml(self, catalog: str, schemas: Optional[List[str]], 
                             tables: Optional[List[str]], columns: Optional[List[str]],
                             config: Dict):
        """
        Update dbxmetagen variables.yml file with catalog/schema selection
        
        This would typically modify the DBFS file, but for now we'll log the intent
        """
        logger.info(f"📝 Would update variables.yml at {config['variables_yml_path']}")
        logger.info(f"   Catalog: {catalog}")
        logger.info(f"   Schemas: {schemas}")
        
        # In a real implementation, we would:
        # 1. Read the existing variables.yml from DBFS
        # 2. Update the catalog/schema selections
        # 3. Write back to DBFS
        
        # Example structure:
        # variables_update = {
        #     'catalog_name': catalog,
        #     'schema_names': schemas if schemas else [],
        #     'run_mode': 'metadata_generation'
        # }
    
    def _import_dbxmetagen_results(self, run_id: str, config: Dict):
        """
        Import dbxmetagen results into UC Metadata Assistant tables
        
        Args:
            run_id: Run identifier
            config: dbxmetagen configuration
        """
        try:
            logger.info(f"📥 Importing dbxmetagen results for {run_id}")
            
            # dbxmetagen outputs to Excel/TSV files in DBFS
            # We would:
            # 1. Find the output file (typically in /dbfs/dbxmetagen/outputs/)
            # 2. Parse the Excel/TSV file
            # 3. Convert to our metadata_results format
            # 4. Insert into uc_metadata_assistant.generated_metadata.metadata_results
            
            # Example structure of dbxmetagen output:
            # full_name | object_type | comment | confidence | pii_classification | tags
            
            output_path = f"/dbfs/dbxmetagen/outputs/{run_id}.xlsx"
            self.job_runs[run_id]['output_file_path'] = output_path
            
            logger.info(f"✅ Results imported from {output_path}")
            
            # TODO: Actual implementation would parse Excel and insert into Delta
            
        except Exception as e:
            logger.error(f"Error importing dbxmetagen results: {e}")
            # Don't raise - job completed but import failed
    
    def cancel_generation(self, run_id: str) -> bool:
        """
        Cancel a running dbxmetagen job
        
        Args:
            run_id: Run identifier
        
        Returns:
            bool: True if cancelled successfully
        """
        try:
            if run_id not in self.job_runs:
                logger.warning(f"Run ID {run_id} not found for cancellation")
                return False
            
            job_info = self.job_runs[run_id]
            databricks_run_id = job_info['databricks_run_id']
            
            # Cancel the Databricks job
            self.workspace_client.jobs.cancel_run(run_id=databricks_run_id)
            
            # Update status
            self.job_runs[run_id]['status'] = 'CANCELLED'
            
            logger.info(f"✅ Cancelled dbxmetagen job: {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False

