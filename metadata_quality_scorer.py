"""
Metadata Quality Scorer
Assesses the quality of existing metadata descriptions
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score breakdown for a metadata description"""
    overall_score: int  # 0-100
    length_score: int  # 0-100
    completeness_score: int  # 0-100
    grammar_score: int  # 0-100
    professionalism_score: int  # 0-100
    informativeness_score: int  # 0-100
    issues: List[str]  # List of specific issues found
    category: str  # 'poor', 'marginal', 'good'


class MetadataQualityScorer:
    """Analyzes and scores metadata description quality"""
    
    def __init__(self):
        # Generic/placeholder terms that indicate poor quality
        self.generic_terms = [
            'data', 'table', 'column', 'field', 'value', 'information',
            'details', 'record', 'entry', 'item', 'object', 'element',
            'description for', 'metadata for', 'contains', 'stores',
            'tbd', 'todo', 'fix', 'update', 'change', 'edit', 'temp'
        ]
        
        # Common placeholder patterns
        self.placeholder_patterns = [
            r'^description\s+for\s+',
            r'^table\s+\d+',
            r'^column\s+\d+',
            r'^schema\s+\d+',
            r'^\w+\s+table$',
            r'^\w+\s+column$',
            r'^n/a$',
            r'^tbd$',
            r'^todo$',
            r'^test$'
        ]
    
    def score_description(self, description: str, object_type: str = 'table') -> QualityScore:
        """
        Score a single description across multiple quality dimensions
        
        Args:
            description: The metadata description to score
            object_type: Type of object (schema, table, column)
        
        Returns:
            QualityScore object with detailed breakdown
        """
        if not description or not description.strip():
            return QualityScore(
                overall_score=0,
                length_score=0,
                completeness_score=0,
                grammar_score=0,
                professionalism_score=0,
                informativeness_score=0,
                issues=['Description is empty'],
                category='poor'
            )
        
        description = description.strip()
        issues = []
        
        # 1. Length Score
        length_score = self._score_length(description, object_type, issues)
        
        # 2. Completeness Score
        completeness_score = self._score_completeness(description, object_type, issues)
        
        # 3. Grammar Score
        grammar_score = self._score_grammar(description, issues)
        
        # 4. Professionalism Score
        professionalism_score = self._score_professionalism(description, issues)
        
        # 5. Informativeness Score
        informativeness_score = self._score_informativeness(description, issues)
        
        # Calculate overall score (weighted average)
        overall_score = int(
            length_score * 0.15 +
            completeness_score * 0.25 +
            grammar_score * 0.15 +
            professionalism_score * 0.20 +
            informativeness_score * 0.25
        )
        
        # Hard caps for very poor quality
        words = description.split()
        word_count = len(words)
        
        # Very short descriptions (≤5 words) are automatically poor, regardless of other scores
        if word_count <= 5:
            overall_score = min(overall_score, 40)
        
        # Categorize
        if overall_score < 41:
            category = 'poor'
        elif overall_score < 71:
            category = 'marginal'
        else:
            category = 'good'
        
        return QualityScore(
            overall_score=overall_score,
            length_score=length_score,
            completeness_score=completeness_score,
            grammar_score=grammar_score,
            professionalism_score=professionalism_score,
            informativeness_score=informativeness_score,
            issues=issues,
            category=category
        )
    
    def _score_length(self, description: str, object_type: str, issues: List[str]) -> int:
        """Score based on description length (words)"""
        words = description.split()
        word_count = len(words)
        
        # Expected word count ranges by object type
        if object_type == 'schema':
            min_good, max_good = 15, 100
            min_marginal = 8
        elif object_type == 'table':
            min_good, max_good = 20, 120
            min_marginal = 10
        else:  # column
            min_good, max_good = 10, 80
            min_marginal = 5
        
        if word_count < 3:
            issues.append(f'Very short description ({word_count} words)')
            return 10
        elif word_count < min_marginal:
            issues.append(f'Short description ({word_count} words)')
            return 40
        elif word_count < min_good:
            return 70
        elif word_count <= max_good:
            return 100
        else:
            issues.append(f'Very long description ({word_count} words)')
            return 80
    
    def _score_completeness(self, description: str, object_type: str, issues: List[str]) -> int:
        """Score based on completeness of information"""
        score = 50  # Base score
        
        # Check for key information elements
        has_business_context = any(term in description.lower() for term in [
            'customer', 'transaction', 'order', 'product', 'sales', 'revenue',
            'user', 'account', 'payment', 'inventory', 'shipping', 'analytics'
        ])
        
        has_technical_detail = any(term in description.lower() for term in [
            'integer', 'string', 'timestamp', 'decimal', 'boolean', 'date',
            'primary key', 'foreign key', 'unique', 'nullable', 'required',
            'calculated', 'derived', 'aggregated'
        ])
        
        has_purpose = any(term in description.lower() for term in [
            'used to', 'used for', 'represents', 'indicates', 'tracks',
            'stores', 'contains', 'records', 'identifies', 'measures'
        ])
        
        # Multiple sentences indicate more complete description
        sentence_count = len([s for s in description.split('.') if s.strip()])
        
        if has_business_context:
            score += 15
        if has_technical_detail:
            score += 15
        if has_purpose:
            score += 10
        if sentence_count >= 2:
            score += 10
        
        if score < 70:
            issues.append('Lacks sufficient detail or context')
        
        return min(score, 100)
    
    def _score_grammar(self, description: str, issues: List[str]) -> int:
        """Score based on grammar and sentence structure"""
        score = 100
        
        # Check for basic grammar issues
        
        # Starts with capital letter
        if not description[0].isupper():
            issues.append('Does not start with capital letter')
            score -= 15
        
        # Ends with proper punctuation
        if not description.rstrip()[-1] in '.!?':
            issues.append('Missing ending punctuation')
            score -= 10
        
        # Has at least one sentence structure
        if '.' not in description and '!' not in description and '?' not in description:
            if len(description.split()) > 5:
                issues.append('No sentence structure (missing periods)')
                score -= 15
        
        # Check for lowercase sentence starts after periods
        sentences = [s.strip() for s in description.split('.') if s.strip()]
        for i, sent in enumerate(sentences[1:], 1):  # Skip first sentence
            if sent and sent[0].islower():
                issues.append('Sentence does not start with capital')
                score -= 10
                break
        
        return max(score, 0)
    
    def _score_professionalism(self, description: str, issues: List[str]) -> int:
        """Score based on professional presentation"""
        score = 100
        
        # Check for ALL CAPS (more than 30% of letters)
        letters = [c for c in description if c.isalpha()]
        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > 0.3:
                issues.append('Excessive capitalization')
                score -= 25
        
        # Check for excessive punctuation
        if '!!' in description or '??' in description or '...' in description:
            issues.append('Excessive punctuation')
            score -= 15
        
        # Check for informal language
        informal_terms = ['gonna', 'wanna', 'kinda', 'sorta', 'yeah', 'nope', 'yup']
        if any(term in description.lower() for term in informal_terms):
            issues.append('Contains informal language')
            score -= 20
        
        # Check for placeholder-like text
        for pattern in self.placeholder_patterns:
            if re.match(pattern, description.lower()):
                issues.append('Appears to be placeholder text')
                score -= 40
                break
        
        return max(score, 0)
    
    def _score_informativeness(self, description: str, issues: List[str]) -> int:
        """Score based on how informative the description is"""
        score = 50  # Base score
        
        description_lower = description.lower()
        
        # Check if description is too generic
        generic_count = sum(1 for term in self.generic_terms if term in description_lower)
        
        if generic_count >= 3:
            issues.append('Very generic description')
            score -= 30
        elif generic_count >= 2:
            issues.append('Somewhat generic description')
            score -= 15
        
        # Check for specific, meaningful terms (domain-specific language)
        # More unique/specific words = higher score
        words = description_lower.split()
        unique_meaningful_words = [
            w for w in words 
            if len(w) > 4 and w not in self.generic_terms and w.isalpha()
        ]
        
        if len(unique_meaningful_words) >= 5:
            score += 30
        elif len(unique_meaningful_words) >= 3:
            score += 20
        elif len(unique_meaningful_words) >= 1:
            score += 10
        else:
            issues.append('Lacks specific, meaningful terms')
        
        # Check if it's just a restatement of the column/table name
        # e.g., "customer_id" → "customer id"
        words_normalized = set(''.join(c for c in description_lower if c.isalnum() or c.isspace()).split())
        if len(words_normalized) <= 3:
            issues.append('Description merely restates the object name')
            score -= 20
        
        return max(min(score, 100), 0)
    
    def assess_catalog_quality(self, unity_service, catalog_name: str, 
                               selected_schemas: List[str] = None,
                               selected_tables: List[str] = None,
                               selected_columns: List[str] = None) -> Dict:
        """
        Scan catalog and assess quality of all described objects
        
        Args:
            unity_service: UnityMetadataService instance
            catalog_name: Catalog to scan
            selected_schemas: Optional list of schema names to include
            selected_tables: Optional list of table full names to include
            selected_columns: Optional list of column full names to include
        
        Returns:
            {
                'assessments': [{full_name, object_type, current_description, quality_score, issues, category}, ...],
                'summary': {
                    'total': int,
                    'poor_count': int,
                    'marginal_count': int,
                    'good_count': int,
                    'avg_score': float
                }
            }
        """
        assessments = []
        
        try:
            # Get all described objects in catalog
            from metadata_copy_utils import MetadataCopyUtils
            copy_utils = MetadataCopyUtils(unity_service)
            
            described_objects = copy_utils.get_objects_with_descriptions(catalog_name)
            
            # Filter by selection
            all_objects = []
            
            # Add selected schemas (if None, include all; if empty list, include none; if has items, include only those)
            if selected_schemas is None:
                # No schema filter specified - include all schemas
                for obj in described_objects['schemas']:
                    obj['object_type'] = 'schema'
                    all_objects.append(obj)
            elif len(selected_schemas) > 0:
                # Specific schemas selected - include only those
                for obj in described_objects['schemas']:
                    if obj['full_name'] in selected_schemas:
                        obj['object_type'] = 'schema'
                        all_objects.append(obj)
            # else: empty list means user selected none - include none
            
            # Add selected tables
            if selected_tables is None:
                # No table filter specified - include all tables
                for obj in described_objects['tables']:
                    obj['object_type'] = 'table'
                    all_objects.append(obj)
            elif len(selected_tables) > 0:
                # Specific tables selected - include only those
                for obj in described_objects['tables']:
                    if obj['full_name'] in selected_tables:
                        obj['object_type'] = 'table'
                        all_objects.append(obj)
            # else: empty list means user selected none - include none
            
            # Add selected columns
            if selected_columns is None:
                # No column filter specified - include all columns
                for obj in described_objects['columns']:
                    obj['object_type'] = 'column'
                    all_objects.append(obj)
            elif len(selected_columns) > 0:
                # Specific columns selected - include only those
                for obj in described_objects['columns']:
                    if obj['full_name'] in selected_columns:
                        obj['object_type'] = 'column'
                        all_objects.append(obj)
            # else: empty list means user selected none - include none
            
            # Score each description
            poor_count = 0
            marginal_count = 0
            good_count = 0
            total_score = 0
            
            for obj in all_objects:
                quality = self.score_description(obj['description'], obj['object_type'])
                
                assessments.append({
                    'full_name': obj['full_name'],
                    'object_type': obj['object_type'],
                    'current_description': obj['description'],
                    'quality_score': quality.overall_score,
                    'length_score': quality.length_score,
                    'completeness_score': quality.completeness_score,
                    'grammar_score': quality.grammar_score,
                    'professionalism_score': quality.professionalism_score,
                    'informativeness_score': quality.informativeness_score,
                    'issues': quality.issues,
                    'category': quality.category
                })
                
                if quality.category == 'poor':
                    poor_count += 1
                elif quality.category == 'marginal':
                    marginal_count += 1
                else:
                    good_count += 1
                
                total_score += quality.overall_score
            
            avg_score = total_score / len(assessments) if assessments else 0
            
            logger.info(f"📊 Quality assessment: {len(assessments)} objects - "
                       f"Poor: {poor_count}, Marginal: {marginal_count}, Good: {good_count}, "
                       f"Avg Score: {avg_score:.1f}")
            
            return {
                'assessments': assessments,
                'summary': {
                    'total': len(assessments),
                    'poor_count': poor_count,
                    'marginal_count': marginal_count,
                    'good_count': good_count,
                    'avg_score': round(avg_score, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing catalog quality: {e}")
            raise

