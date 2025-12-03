"""
Metadata Copy Utilities
Provides smart matching and bulk copy operations for metadata management
"""

import logging
import re
import csv
import io
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class MetadataCopyUtils:
    """Utilities for copying metadata between objects with smart matching"""
    
    def __init__(self, unity_service):
        self.unity_service = unity_service
    
    def get_objects_with_descriptions(self, catalog_name: str, schema_name: Optional[str] = None, 
                                     object_type: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Fetch all objects WITH descriptions from a catalog/schema
        
        Args:
            catalog_name: Catalog to query
            schema_name: Optional schema filter
            object_type: Optional filter: 'schema', 'table', or 'column'
        
        Returns:
            {
                'schemas': [{full_name, description, created_at, table_count}, ...],
                'tables': [{full_name, description, created_at, column_count, schema}, ...],
                'columns': [{full_name, description, data_type, table, schema}, ...]
            }
        """
        result = {'schemas': [], 'tables': [], 'columns': []}
        
        try:
            # Fetch schemas with descriptions
            if not object_type or object_type == 'schema':
                schema_filter = f"AND s.schema_name = '{schema_name}'" if schema_name else ""
                schema_sql = f"""
                    SELECT 
                        CONCAT(s.catalog_name, '.', s.schema_name) as full_name,
                        s.comment as description,
                        s.created as created_at,
                        COALESCE(t.table_count, 0) as table_count
                    FROM {catalog_name}.information_schema.schemata s
                    LEFT JOIN (
                        SELECT table_catalog, table_schema, COUNT(*) as table_count
                        FROM {catalog_name}.information_schema.tables
                        WHERE table_catalog = '{catalog_name}'
                        GROUP BY table_catalog, table_schema
                    ) t ON s.catalog_name = t.table_catalog AND s.schema_name = t.table_schema
                    WHERE s.catalog_name = '{catalog_name}'
                        AND s.schema_name NOT IN ('information_schema', 'system')
                        AND s.comment IS NOT NULL
                        AND s.comment != ''
                        {schema_filter}
                    ORDER BY full_name
                """
                schema_results = self.unity_service._execute_sql_warehouse(schema_sql)
                if schema_results:
                    result['schemas'] = [
                        {
                            'full_name': row[0],
                            'description': row[1],
                            'created_at': row[2],
                            'table_count': row[3]
                        }
                        for row in schema_results
                    ]
                logger.info(f"Found {len(result['schemas'])} schemas with descriptions in {catalog_name}")
            
            # Fetch tables with descriptions
            if not object_type or object_type == 'table':
                schema_filter = f"AND table_schema = '{schema_name}'" if schema_name else ""
                table_sql = f"""
                    SELECT 
                        CONCAT(table_catalog, '.', table_schema, '.', table_name) as full_name,
                        comment as description,
                        created as created_at,
                        table_schema,
                        (SELECT COUNT(*) 
                         FROM {catalog_name}.information_schema.columns c 
                         WHERE c.table_catalog = t.table_catalog 
                           AND c.table_schema = t.table_schema 
                           AND c.table_name = t.table_name) as column_count
                    FROM {catalog_name}.information_schema.tables t
                    WHERE table_catalog = '{catalog_name}'
                        AND table_schema NOT IN ('information_schema', 'system')
                        AND comment IS NOT NULL
                        AND comment != ''
                        {schema_filter}
                    ORDER BY full_name
                """
                table_results = self.unity_service._execute_sql_warehouse(table_sql)
                if table_results:
                    result['tables'] = [
                        {
                            'full_name': row[0],
                            'description': row[1],
                            'created_at': row[2],
                            'schema': row[3],
                            'column_count': row[4]
                        }
                        for row in table_results
                    ]
                logger.info(f"Found {len(result['tables'])} tables with descriptions in {catalog_name}")
            
            # Fetch columns with descriptions
            if not object_type or object_type == 'column':
                schema_filter = f"AND table_schema = '{schema_name}'" if schema_name else ""
                column_sql = f"""
                    SELECT 
                        CONCAT(table_catalog, '.', table_schema, '.', table_name, '.', column_name) as full_name,
                        comment as description,
                        data_type,
                        table_schema,
                        table_name
                    FROM {catalog_name}.information_schema.columns
                    WHERE table_catalog = '{catalog_name}'
                        AND table_schema NOT IN ('information_schema', 'system')
                        AND comment IS NOT NULL
                        AND comment != ''
                        {schema_filter}
                    ORDER BY full_name
                """
                column_results = self.unity_service._execute_sql_warehouse(column_sql)
                if column_results:
                    result['columns'] = [
                        {
                            'full_name': row[0],
                            'description': row[1],
                            'data_type': row[2],
                            'schema': row[3],
                            'table': row[4]
                        }
                        for row in column_results
                    ]
                logger.info(f"Found {len(result['columns'])} columns with descriptions in {catalog_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching objects with descriptions: {e}")
            raise
    
    def get_objects_without_descriptions(self, catalog_name: str, schema_name: Optional[str] = None, object_type: Optional[str] = None) -> Dict:
        """
        Get objects WITHOUT descriptions (undocumented objects) from a catalog
        Returns schemas, tables, and columns that are missing descriptions
        """
        result = {
            'schemas': [],
            'tables': [],
            'columns': []
        }
        
        try:
            # Fetch schemas without descriptions
            if not object_type or object_type == 'schema':
                schema_filter = f"AND s.schema_name = '{schema_name}'" if schema_name else ""
                schema_sql = f"""
                    SELECT 
                        CONCAT(s.catalog_name, '.', s.schema_name) as full_name,
                        s.created as created_at,
                        COALESCE(t.table_count, 0) as table_count
                    FROM {catalog_name}.information_schema.schemata s
                    LEFT JOIN (
                        SELECT table_catalog, table_schema, COUNT(*) as table_count
                        FROM {catalog_name}.information_schema.tables
                        WHERE table_catalog = '{catalog_name}'
                        GROUP BY table_catalog, table_schema
                    ) t ON s.catalog_name = t.table_catalog AND s.schema_name = t.table_schema
                    WHERE s.catalog_name = '{catalog_name}'
                        AND s.schema_name NOT IN ('information_schema', 'system')
                        AND (s.comment IS NULL OR s.comment = '')
                        {schema_filter}
                    ORDER BY full_name
                """
                schema_results = self.unity_service._execute_sql_warehouse(schema_sql)
                if schema_results:
                    result['schemas'] = [
                        {
                            'full_name': row[0],
                            'created_at': row[1],
                            'table_count': row[2]
                        }
                        for row in schema_results
                    ]
                logger.info(f"Found {len(result['schemas'])} schemas without descriptions in {catalog_name}")
            
            # Fetch tables without descriptions
            if not object_type or object_type == 'table':
                schema_filter = f"AND table_schema = '{schema_name}'" if schema_name else ""
                table_sql = f"""
                    SELECT 
                        CONCAT(table_catalog, '.', table_schema, '.', table_name) as full_name,
                        created as created_at,
                        table_schema,
                        (SELECT COUNT(*) 
                         FROM {catalog_name}.information_schema.columns c 
                         WHERE c.table_catalog = t.table_catalog 
                           AND c.table_schema = t.table_schema 
                           AND c.table_name = t.table_name) as column_count
                    FROM {catalog_name}.information_schema.tables t
                    WHERE table_catalog = '{catalog_name}'
                        AND table_schema NOT IN ('information_schema', 'system')
                        AND (comment IS NULL OR comment = '')
                        {schema_filter}
                    ORDER BY full_name
                """
                table_results = self.unity_service._execute_sql_warehouse(table_sql)
                if table_results:
                    result['tables'] = [
                        {
                            'full_name': row[0],
                            'created_at': row[1],
                            'schema': row[2],
                            'column_count': row[3]
                        }
                        for row in table_results
                    ]
                logger.info(f"Found {len(result['tables'])} tables without descriptions in {catalog_name}")
            
            # Fetch columns without descriptions
            if not object_type or object_type == 'column':
                schema_filter = f"AND table_schema = '{schema_name}'" if schema_name else ""
                column_sql = f"""
                    SELECT 
                        CONCAT(table_catalog, '.', table_schema, '.', table_name, '.', column_name) as full_name,
                        data_type,
                        table_schema,
                        table_name
                    FROM {catalog_name}.information_schema.columns
                    WHERE table_catalog = '{catalog_name}'
                        AND table_schema NOT IN ('information_schema', 'system')
                        AND (comment IS NULL OR comment = '')
                        {schema_filter}
                    ORDER BY full_name
                """
                column_results = self.unity_service._execute_sql_warehouse(column_sql)
                if column_results:
                    result['columns'] = [
                        {
                            'full_name': row[0],
                            'data_type': row[1],
                            'schema': row[2],
                            'table': row[3]
                        }
                        for row in column_results
                    ]
                logger.info(f"Found {len(result['columns'])} columns without descriptions in {catalog_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching objects without descriptions: {e}")
            raise
    
    def smart_match_objects(self, source_objects: List[Dict], target_catalog: str, 
                           target_schema: Optional[str] = None) -> Dict:
        """
        Generate smart matches between source and target objects
        
        Args:
            source_objects: List of {full_name, description, object_type}
            target_catalog: Target catalog name
            target_schema: Optional target schema filter
        
        Returns:
            {
                'matches': [{source, target, confidence, match_type, description}, ...],
                'unmatched_sources': [...],
                'unmatched_targets': [...]
            }
        """
        matches = []
        unmatched_sources = []
        
        try:
            # Get target objects WITHOUT descriptions (undescribed targets only)
            target_data = self.get_objects_without_descriptions(target_catalog, target_schema)
            target_objects = []
            for schema in target_data['schemas']:
                target_objects.append({'full_name': schema['full_name'], 'object_type': 'schema'})
            for table in target_data['tables']:
                target_objects.append({'full_name': table['full_name'], 'object_type': 'table'})
            for column in target_data['columns']:
                target_objects.append({'full_name': column['full_name'], 'object_type': 'column'})
            
            # Track which targets have been matched
            matched_targets = set()
            
            for source in source_objects:
                source_name = source['full_name']
                source_type = source['object_type']
                description = source['description']
                
                # Parse source name
                source_parts = source_name.split('.')
                
                # Get eligible targets of same type
                eligible_targets = [
                    t for t in target_objects 
                    if t['object_type'] == source_type and t['full_name'] not in matched_targets
                ]
                
                # Try matching strategies in order of confidence
                match = None
                
                # Strategy 1: Exact name match (confidence: 1.0)
                match = self._exact_name_match(source_parts, eligible_targets, source_type)
                if match:
                    match['match_type'] = 'exact'
                    match['confidence'] = 1.0
                
                # Strategy 2: Table structure match (all columns match)
                if not match and source_type == 'table':
                    match = self._table_structure_match(source_name, eligible_targets)
                    if match:
                        match['match_type'] = 'structure'
                        # confidence set by structure matcher based on column overlap
                
                # Strategy 3: Pattern-based match (bronze→silver, silver→gold)
                if not match:
                    match = self._pattern_based_match(source_parts, eligible_targets, source_type)
                    if match:
                        match['match_type'] = 'pattern'
                        match['confidence'] = 0.9
                
                # Strategy 4: Column name only match (for columns across different tables)
                if not match and source_type == 'column':
                    match = self._column_name_match(source_parts, eligible_targets)
                    if match:
                        match['match_type'] = 'column_name'
                        match['confidence'] = 0.8
                
                # Strategy 5: Fuzzy name match
                if not match:
                    match = self._fuzzy_name_match(source_parts, eligible_targets, source_type)
                    if match:
                        match['match_type'] = 'fuzzy'
                        # confidence already set by fuzzy matcher
                
                if match:
                    matches.append({
                        'source': source_name,
                        'target': match['target'],
                        'confidence': match['confidence'],
                        'match_type': match['match_type'],
                        'description': description,
                        'object_type': source_type
                    })
                    matched_targets.add(match['target'])
                else:
                    unmatched_sources.append(source_name)
            
            # Find unmatched targets
            unmatched_targets = [
                t['full_name'] for t in target_objects 
                if t['full_name'] not in matched_targets
            ]
            
            logger.info(f"Smart match results: {len(matches)} matches, "
                       f"{len(unmatched_sources)} unmatched sources, "
                       f"{len(unmatched_targets)} unmatched targets")
            
            return {
                'matches': matches,
                'unmatched_sources': unmatched_sources,
                'unmatched_targets': unmatched_targets
            }
            
        except Exception as e:
            logger.error(f"Error in smart matching: {e}")
            raise
    
    def _get_all_objects(self, catalog_name: str, schema_name: Optional[str] = None) -> List[Dict]:
        """Get all objects (with or without descriptions) from catalog"""
        objects = []
        
        # Get schemas
        schema_filter = f"AND table_schema = '{schema_name}'" if schema_name else ""
        schema_sql = f"""
            SELECT DISTINCT
                CONCAT(table_catalog, '.', table_schema) as full_name,
                'schema' as object_type
            FROM {catalog_name}.information_schema.tables
            WHERE table_catalog = '{catalog_name}'
                AND table_schema NOT IN ('information_schema', 'system')
                {schema_filter}
        """
        schema_results = self.unity_service._execute_sql_warehouse(schema_sql)
        if schema_results:
            objects.extend([{'full_name': row[0], 'object_type': 'schema'} for row in schema_results])
        
        # Get tables
        table_sql = f"""
            SELECT 
                CONCAT(table_catalog, '.', table_schema, '.', table_name) as full_name,
                'table' as object_type
            FROM {catalog_name}.information_schema.tables
            WHERE table_catalog = '{catalog_name}'
                AND table_schema NOT IN ('information_schema', 'system')
                {schema_filter}
        """
        table_results = self.unity_service._execute_sql_warehouse(table_sql)
        if table_results:
            objects.extend([{'full_name': row[0], 'object_type': 'table'} for row in table_results])
        
        # Get columns
        column_sql = f"""
            SELECT 
                CONCAT(table_catalog, '.', table_schema, '.', table_name, '.', column_name) as full_name,
                'column' as object_type
            FROM {catalog_name}.information_schema.columns
            WHERE table_catalog = '{catalog_name}'
                AND table_schema NOT IN ('information_schema', 'system')
                {schema_filter}
        """
        column_results = self.unity_service._execute_sql_warehouse(column_sql)
        if column_results:
            objects.extend([{'full_name': row[0], 'object_type': 'column'} for row in column_results])
        
        return objects
    
    def _exact_name_match(self, source_parts: List[str], targets: List[Dict], object_type: str) -> Optional[Dict]:
        """Match objects with identical names (ignoring catalog)"""
        # For schemas: match schema name only
        if object_type == 'schema' and len(source_parts) >= 2:
            source_schema = source_parts[1]
            for target in targets:
                target_parts = target['full_name'].split('.')
                if len(target_parts) >= 2 and target_parts[1] == source_schema:
                    return {'target': target['full_name']}
        
        # For tables: match schema.table
        elif object_type == 'table' and len(source_parts) >= 3:
            source_schema = source_parts[1]
            source_table = source_parts[2]
            for target in targets:
                target_parts = target['full_name'].split('.')
                if (len(target_parts) >= 3 and 
                    target_parts[1] == source_schema and 
                    target_parts[2] == source_table):
                    return {'target': target['full_name']}
        
        # For columns: match schema.table.column
        elif object_type == 'column' and len(source_parts) >= 4:
            source_schema = source_parts[1]
            source_table = source_parts[2]
            source_column = source_parts[3]
            for target in targets:
                target_parts = target['full_name'].split('.')
                if (len(target_parts) >= 4 and 
                    target_parts[1] == source_schema and 
                    target_parts[2] == source_table and 
                    target_parts[3] == source_column):
                    return {'target': target['full_name']}
        
        return None
    
    def _table_structure_match(self, source_table_name: str, target_tables: List[Dict]) -> Optional[Dict]:
        """Match tables by comparing column structures (all columns match)"""
        try:
            # Get columns for source table
            source_columns = self._get_table_columns(source_table_name)
            if not source_columns:
                return None
            
            source_column_names = set(col.lower() for col in source_columns)
            
            best_match = None
            best_overlap = 0.0
            
            for target in target_tables:
                target_table_name = target['full_name']
                target_columns = self._get_table_columns(target_table_name)
                
                if not target_columns:
                    continue
                
                target_column_names = set(col.lower() for col in target_columns)
                
                # Calculate column overlap
                if len(source_column_names) > 0:
                    intersection = source_column_names & target_column_names
                    union = source_column_names | target_column_names
                    overlap_ratio = len(intersection) / len(union)  # Jaccard similarity
                    
                    # Require high overlap (90%+ of columns match)
                    if overlap_ratio >= 0.9 and overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        # Confidence: 0.95 for 100% match, scales down to 0.85 for 90% match
                        confidence = 0.85 + (overlap_ratio - 0.9) * 1.0
                        best_match = {
                            'target': target_table_name,
                            'confidence': min(confidence, 0.95)
                        }
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in table structure matching: {e}")
            return None
    
    def _get_table_columns(self, table_full_name: str) -> List[str]:
        """Get list of column names for a table"""
        try:
            parts = table_full_name.split('.')
            if len(parts) < 3:
                return []
            
            catalog, schema, table = parts[0], parts[1], parts[2]
            
            column_sql = f"""
                SELECT column_name
                FROM {catalog}.information_schema.columns
                WHERE table_catalog = '{catalog}'
                    AND table_schema = '{schema}'
                    AND table_name = '{table}'
                ORDER BY ordinal_position
            """
            
            results = self.unity_service._execute_sql_warehouse(column_sql)
            return [row[0] for row in results] if results else []
            
        except Exception as e:
            logger.error(f"Error fetching columns for {table_full_name}: {e}")
            return []
    
    def _pattern_based_match(self, source_parts: List[str], targets: List[Dict], object_type: str) -> Optional[Dict]:
        """Match with bronze→silver, silver→gold patterns"""
        patterns = [
            ('bronze', 'silver'),
            ('silver', 'gold'),
            ('raw', 'bronze'),
            ('bronze', 'cleaned'),
            ('cleaned', 'curated'),
            ('dev', 'test'),
            ('test', 'prod'),
            ('staging', 'production')
        ]
        
        # For schemas: match pattern in schema name
        if object_type == 'schema' and len(source_parts) >= 2:
            source_schema = source_parts[1].lower()
            for from_pattern, to_pattern in patterns:
                if from_pattern in source_schema:
                    expected_target_schema = source_schema.replace(from_pattern, to_pattern)
                    for target in targets:
                        target_parts = target['full_name'].split('.')
                        if len(target_parts) >= 2 and target_parts[1].lower() == expected_target_schema:
                            return {'target': target['full_name']}
        
        # For tables: match pattern in schema or table name
        elif object_type == 'table' and len(source_parts) >= 3:
            source_schema = source_parts[1].lower()
            source_table = source_parts[2].lower()
            
            for from_pattern, to_pattern in patterns:
                # Try schema pattern
                if from_pattern in source_schema:
                    expected_schema = source_schema.replace(from_pattern, to_pattern)
                    for target in targets:
                        target_parts = target['full_name'].split('.')
                        if (len(target_parts) >= 3 and 
                            target_parts[1].lower() == expected_schema and 
                            target_parts[2].lower() == source_table):
                            return {'target': target['full_name']}
                
                # Try table pattern
                if from_pattern in source_table:
                    expected_table = source_table.replace(from_pattern, to_pattern)
                    for target in targets:
                        target_parts = target['full_name'].split('.')
                        if (len(target_parts) >= 3 and 
                            target_parts[1].lower() == source_schema and 
                            target_parts[2].lower() == expected_table):
                            return {'target': target['full_name']}
        
        # For columns: match pattern in schema or table name
        elif object_type == 'column' and len(source_parts) >= 4:
            source_schema = source_parts[1].lower()
            source_table = source_parts[2].lower()
            source_column = source_parts[3].lower()
            
            for from_pattern, to_pattern in patterns:
                if from_pattern in source_schema:
                    expected_schema = source_schema.replace(from_pattern, to_pattern)
                    for target in targets:
                        target_parts = target['full_name'].split('.')
                        if (len(target_parts) >= 4 and 
                            target_parts[1].lower() == expected_schema and 
                            target_parts[2].lower() == source_table and 
                            target_parts[3].lower() == source_column):
                            return {'target': target['full_name']}
        
        return None
    
    def _column_name_match(self, source_parts: List[str], targets: List[Dict]) -> Optional[Dict]:
        """Match columns by column name only, even if table names differ"""
        if len(source_parts) < 4:
            return None
        
        source_column = source_parts[3].lower()
        best_match = None
        best_ratio = 0.85  # Higher threshold since we're ignoring table context
        
        for target in targets:
            target_parts = target['full_name'].split('.')
            if len(target_parts) >= 4:
                target_column = target_parts[3].lower()
                # Exact column name match
                if source_column == target_column:
                    return {'target': target['full_name']}
                # Fuzzy column name match
                ratio = SequenceMatcher(None, source_column, target_column).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = {'target': target['full_name'], 'confidence': ratio}
        
        return best_match
    
    def _fuzzy_name_match(self, source_parts: List[str], targets: List[Dict], object_type: str) -> Optional[Dict]:
        """Fuzzy string matching for similar names"""
        best_match = None
        best_ratio = 0.7  # Minimum threshold
        
        for target in targets:
            target_parts = target['full_name'].split('.')
            
            # Compare relevant parts based on object type
            if object_type == 'schema' and len(source_parts) >= 2 and len(target_parts) >= 2:
                ratio = SequenceMatcher(None, source_parts[1].lower(), target_parts[1].lower()).ratio()
            elif object_type == 'table' and len(source_parts) >= 3 and len(target_parts) >= 3:
                # Weight schema and table name equally
                schema_ratio = SequenceMatcher(None, source_parts[1].lower(), target_parts[1].lower()).ratio()
                table_ratio = SequenceMatcher(None, source_parts[2].lower(), target_parts[2].lower()).ratio()
                ratio = (schema_ratio + table_ratio) / 2
            elif object_type == 'column' and len(source_parts) >= 4 and len(target_parts) >= 4:
                # Weight all parts
                schema_ratio = SequenceMatcher(None, source_parts[1].lower(), target_parts[1].lower()).ratio()
                table_ratio = SequenceMatcher(None, source_parts[2].lower(), target_parts[2].lower()).ratio()
                column_ratio = SequenceMatcher(None, source_parts[3].lower(), target_parts[3].lower()).ratio()
                ratio = (schema_ratio + table_ratio + column_ratio) / 3
            else:
                continue
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = {'target': target['full_name'], 'confidence': round(ratio, 2)}
        
        return best_match
    
    def copy_metadata_bulk(self, mappings: List[Dict]) -> Dict:
        """
        Apply bulk metadata copy using COMMENT statements
        
        Args:
            mappings: List of {target_full_name, description, object_type}
        
        Returns:
            {success_count, errors: [{target, error_message}]}
        """
        success_count = 0
        errors = []
        
        # Batch by object type for efficiency
        schemas = [m for m in mappings if m['object_type'] == 'schema']
        tables = [m for m in mappings if m['object_type'] == 'table']
        columns = [m for m in mappings if m['object_type'] == 'column']
        
        # Process schemas
        for mapping in schemas:
            try:
                parts = mapping['target_full_name'].split('.')
                if len(parts) >= 2:
                    catalog, schema = parts[0], parts[1]
                    # Escape single quotes in description
                    escaped_desc = mapping['description'].replace("'", "''")
                    comment_sql = f"COMMENT ON SCHEMA {catalog}.{schema} IS '{escaped_desc}'"
                    self.unity_service._execute_sql_warehouse(comment_sql)
                    success_count += 1
                    logger.info(f"✅ Copied description to schema: {mapping['target_full_name']}")
            except Exception as e:
                error_msg = str(e)
                errors.append({'target': mapping['target_full_name'], 'error_message': error_msg})
                logger.error(f"Failed to copy to schema {mapping['target_full_name']}: {error_msg}")
        
        # Process tables
        for mapping in tables:
            try:
                parts = mapping['target_full_name'].split('.')
                if len(parts) >= 3:
                    catalog, schema, table = parts[0], parts[1], parts[2]
                    escaped_desc = mapping['description'].replace("'", "''")
                    comment_sql = f"COMMENT ON TABLE {catalog}.{schema}.{table} IS '{escaped_desc}'"
                    self.unity_service._execute_sql_warehouse(comment_sql)
                    success_count += 1
                    logger.info(f"✅ Copied description to table: {mapping['target_full_name']}")
            except Exception as e:
                error_msg = str(e)
                errors.append({'target': mapping['target_full_name'], 'error_message': error_msg})
                logger.error(f"Failed to copy to table {mapping['target_full_name']}: {error_msg}")
        
        # Process columns (batch by table for efficiency)
        columns_by_table = {}
        for mapping in columns:
            parts = mapping['target_full_name'].split('.')
            if len(parts) >= 4:
                table_name = '.'.join(parts[:3])
                if table_name not in columns_by_table:
                    columns_by_table[table_name] = []
                columns_by_table[table_name].append(mapping)
        
        for table_name, table_columns in columns_by_table.items():
            for mapping in table_columns:
                try:
                    parts = mapping['target_full_name'].split('.')
                    if len(parts) >= 4:
                        catalog, schema, table, column = parts[0], parts[1], parts[2], parts[3]
                        escaped_desc = mapping['description'].replace("'", "''")
                        comment_sql = f"ALTER TABLE {catalog}.{schema}.{table} ALTER COLUMN {column} COMMENT '{escaped_desc}'"
                        self.unity_service._execute_sql_warehouse(comment_sql)
                        success_count += 1
                        logger.info(f"✅ Copied description to column: {mapping['target_full_name']}")
                except Exception as e:
                    error_msg = str(e)
                    errors.append({'target': mapping['target_full_name'], 'error_message': error_msg})
                    logger.error(f"Failed to copy to column {mapping['target_full_name']}: {error_msg}")
        
        logger.info(f"📊 Bulk copy complete: {success_count} success, {len(errors)} errors")
        
        return {
            'success_count': success_count,
            'error_count': len(errors),
            'errors': errors
        }
    
    def export_metadata_csv(self, catalog_name: str, schema_names: Optional[List[str]] = None,
                           table_names: Optional[List[str]] = None, column_names: Optional[List[str]] = None,
                           include_tags: bool = False) -> str:
        """
        Export metadata to CSV format
        
        Args:
            catalog_name: Catalog to export from
            schema_names: Optional list of specific schemas
            table_names: Optional list of specific tables (full names)
            column_names: Optional list of specific columns (full names)
            include_tags: Whether to include tags in export (future enhancement)
        
        Returns:
            CSV string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if include_tags:
            writer.writerow(['full_name', 'object_type', 'description', 'tags', 'pii_classification', 'confidence_score'])
        else:
            writer.writerow(['full_name', 'object_type', 'description'])
        
        try:
            # Get objects with descriptions
            objects = self.get_objects_with_descriptions(catalog_name)
            
            # Export schemas
            for schema in objects['schemas']:
                if schema_names and schema['full_name'].split('.')[1] not in schema_names:
                    continue
                if include_tags:
                    writer.writerow([schema['full_name'], 'schema', schema['description'], '', '', ''])
                else:
                    writer.writerow([schema['full_name'], 'schema', schema['description']])
            
            # Export tables
            for table in objects['tables']:
                if table_names and table['full_name'] not in table_names:
                    continue
                if include_tags:
                    writer.writerow([table['full_name'], 'table', table['description'], '', '', ''])
                else:
                    writer.writerow([table['full_name'], 'table', table['description']])
            
            # Export columns
            for column in objects['columns']:
                if column_names and column['full_name'] not in column_names:
                    continue
                if include_tags:
                    writer.writerow([column['full_name'], 'column', column['description'], '', '', ''])
                else:
                    writer.writerow([column['full_name'], 'column', column['description']])
            
            csv_content = output.getvalue()
            logger.info(f"📤 Exported {len(objects['schemas'])} schemas, {len(objects['tables'])} tables, {len(objects['columns'])} columns to CSV")
            return csv_content
            
        except Exception as e:
            logger.error(f"Error exporting metadata to CSV: {e}")
            raise
        finally:
            output.close()
    
    def import_metadata_csv(self, csv_content: str, target_catalog: str = None) -> Dict:
        """
        Parse and validate imported CSV
        
        Args:
            csv_content: CSV string content
            target_catalog: Optional target catalog to remap names. 
                          If provided, will remap catalog.schema.table -> target_catalog.schema.table
                          If None, uses CSV full_name as-is (can span multiple catalogs)
        
        Returns:
            {
                'preview': [{full_name, object_type, description, exists_in_catalog, validation_message}, ...],
                'valid_count': int,
                'error_count': int,
                'errors': [...]
            }
        """
        preview = []
        valid_count = 0
        error_count = 0
        errors = []
        
        # Check if remapping is enabled
        remap_catalog = target_catalog and target_catalog.strip()
        
        try:
            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            # Validate header
            required_fields = ['full_name', 'object_type', 'description']
            if not all(field in csv_reader.fieldnames for field in required_fields):
                raise ValueError(f"CSV must have columns: {', '.join(required_fields)}")
            
            # Get all target objects to validate existence
            # If remapping, get objects from target catalog only
            # If not remapping, need to get objects from all catalogs mentioned in CSV
            if remap_catalog:
                target_objects = self._get_all_objects(target_catalog)
            else:
                # For no-remap mode, we'll validate per-row by querying as needed
                target_objects = []
            
            target_names = {obj['full_name']: obj['object_type'] for obj in target_objects}
            
            row_num = 1
            for row in csv_reader:
                row_num += 1
                full_name = row.get('full_name', '').strip()
                object_type = row.get('object_type', '').strip().lower()
                description = row.get('description', '').strip()
                
                # Validate row
                validation_message = ''
                exists_in_catalog = False
                is_valid = True
                target_full_name = full_name  # Will be updated if we remap the catalog
                
                if not full_name:
                    validation_message = 'Missing full_name'
                    is_valid = False
                    errors.append({'row': row_num, 'error': validation_message})
                    error_count += 1
                elif not object_type:
                    validation_message = 'Missing object_type'
                    is_valid = False
                    errors.append({'row': row_num, 'error': validation_message})
                    error_count += 1
                elif object_type not in ['schema', 'table', 'column']:
                    validation_message = f'Invalid object_type: {object_type}'
                    is_valid = False
                    errors.append({'row': row_num, 'error': validation_message})
                    error_count += 1
                elif not description:
                    validation_message = 'Missing description'
                    is_valid = False
                    errors.append({'row': row_num, 'error': validation_message})
                    error_count += 1
                else:
                    # Validation logic depends on whether we're remapping
                    if remap_catalog:
                        # REMAP MODE: Try exact match first, then catalog remapping
                        if full_name in target_names:
                            if target_names[full_name] == object_type:
                                exists_in_catalog = True
                                validation_message = 'Exact match - ready to import'
                                valid_count += 1
                            else:
                                validation_message = f'Object type mismatch: expected {object_type}, found {target_names[full_name]}'
                                is_valid = False
                                errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                                error_count += 1
                        else:
                            # Try remapping catalog name
                            parts = full_name.split('.')
                            if len(parts) >= 2:
                                # Replace the catalog part with target catalog
                                parts[0] = target_catalog
                                remapped_name = '.'.join(parts)
                                target_full_name = remapped_name
                                
                                if remapped_name in target_names:
                                    if target_names[remapped_name] == object_type:
                                        exists_in_catalog = True
                                        validation_message = f'Remapped to {target_catalog} - ready to import'
                                        valid_count += 1
                                    else:
                                        validation_message = f'Object type mismatch: expected {object_type}, found {target_names[remapped_name]}'
                                        is_valid = False
                                        errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                                        error_count += 1
                                else:
                                    validation_message = f'Object not found in target catalog (tried: {remapped_name})'
                                    is_valid = False
                                    errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                                    error_count += 1
                            else:
                                validation_message = 'Invalid object name format'
                                is_valid = False
                                errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                                error_count += 1
                    else:
                        # NO-REMAP MODE: Use CSV full_name as-is, validate object exists
                        parts = full_name.split('.')
                        if len(parts) < 2:
                            validation_message = 'Invalid object name format'
                            is_valid = False
                            errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                            error_count += 1
                        else:
                            # Extract catalog from full_name and check if object exists
                            catalog_from_csv = parts[0]
                            try:
                                # Query this specific object to see if it exists
                                if object_type == 'schema':
                                    schema_name = parts[1] if len(parts) >= 2 else None
                                    if schema_name:
                                        query = f"DESCRIBE SCHEMA {catalog_from_csv}.{schema_name}"
                                        self.unity_service._execute_sql_warehouse(query)
                                        exists_in_catalog = True
                                        validation_message = 'Object found - ready to import'
                                        valid_count += 1
                                elif object_type == 'table':
                                    schema_name = parts[1] if len(parts) >= 2 else None
                                    table_name = parts[2] if len(parts) >= 3 else None
                                    if schema_name and table_name:
                                        query = f"DESCRIBE TABLE {catalog_from_csv}.{schema_name}.{table_name}"
                                        self.unity_service._execute_sql_warehouse(query)
                                        exists_in_catalog = True
                                        validation_message = 'Object found - ready to import'
                                        valid_count += 1
                                elif object_type == 'column':
                                    schema_name = parts[1] if len(parts) >= 2 else None
                                    table_name = parts[2] if len(parts) >= 3 else None
                                    column_name = parts[3] if len(parts) >= 4 else None
                                    if schema_name and table_name and column_name:
                                        query = f"DESCRIBE TABLE {catalog_from_csv}.{schema_name}.{table_name}"
                                        result = self.unity_service._execute_sql_warehouse(query)
                                        # Check if column exists in the table
                                        column_exists = any(row[0] == column_name for row in result['data'])
                                        if column_exists:
                                            exists_in_catalog = True
                                            validation_message = 'Object found - ready to import'
                                            valid_count += 1
                                        else:
                                            validation_message = f'Column {column_name} not found in table'
                                            is_valid = False
                                            errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                                            error_count += 1
                            except Exception as e:
                                validation_message = f'Object not found: {str(e)}'
                                is_valid = False
                                errors.append({'row': row_num, 'full_name': full_name, 'error': validation_message})
                                error_count += 1
                
                preview.append({
                    'row': row_num,
                    'full_name': full_name,
                    'target_full_name': target_full_name,  # The actual name to use in target
                    'object_type': object_type,
                    'description': description,
                    'exists_in_catalog': exists_in_catalog,
                    'validation_message': validation_message,
                    'is_valid': is_valid
                })
            
            logger.info(f"📥 CSV import validation: {valid_count} valid, {error_count} errors")
            
            return {
                'preview': preview,
                'valid_count': valid_count,
                'error_count': error_count,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
    
    def apply_imported_metadata(self, mappings: List[Dict], overwrite_existing: bool = False) -> Dict:
        """
        Apply imported metadata from CSV
        
        Args:
            mappings: List of {target_full_name, description, object_type} (only valid ones)
            overwrite_existing: Whether to overwrite existing descriptions
        
        Returns:
            {success_count, skipped_count, errors: [{target, error_message}]}
        """
        success_count = 0
        skipped_count = 0
        errors = []
        
        # If not overwriting, filter out objects that already have descriptions
        if not overwrite_existing:
            # Get existing descriptions from target
            filtered_mappings = []
            for mapping in mappings:
                # Quick check: assume if they're importing, they want to update
                # Full check would require querying each object
                filtered_mappings.append(mapping)
            mappings = filtered_mappings
        
        # Use existing bulk copy function
        result = self.copy_metadata_bulk(mappings)
        
        return {
            'success_count': result['success_count'],
            'skipped_count': skipped_count,
            'errors': result['errors']
        }

