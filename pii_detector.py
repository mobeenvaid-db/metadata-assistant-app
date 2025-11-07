"""
Self-contained PII Detection Module
==================================

Lightweight PII/PHI detection without external dependencies.
Provides enterprise-grade data classification and policy tagging.

Based on common PII patterns and data analysis techniques used in dbxmetagen.

Version: 2.0 (Batch Processing)
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Set, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PIIDetector:
    """
    Self-contained PII detection with pattern matching and data analysis.
    Provides similar functionality to Presidio but embedded and lightweight.
    """
    
    def __init__(self, settings_manager=None, llm_service=None):
        self.settings_manager = settings_manager
        self.llm_service = llm_service
        self.patterns = self._initialize_patterns()
        self.keywords = self._initialize_keywords()
        self.data_classifications = self._initialize_classifications()
        
        # PII detection settings
        self.enabled = True  # PII detection enabled by default
        self.llm_assessment = True if llm_service else False  # LLM-based PII detection if LLM service available
        
        # Get configured LLM model for PII detection
        self.llm_model = self._get_pii_llm_model()
        
        # Load custom patterns if settings manager is available
        if self.settings_manager:
            self._load_custom_patterns()
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize PII detection patterns"""
        return {
            # Personal Identifiers
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'date_of_birth': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            
            # Medical
            'mrn': re.compile(r'\b[A-Z0-9]{6,12}\b'),  # Medical Record Numbers
            'npi': re.compile(r'\b\d{10}\b'),  # National Provider Identifier
            'icd': re.compile(r'\b[A-Z]\d{2}(\.\d+)?\b'),  # ICD codes
            
            # Financial
            'account_number': re.compile(r'\b\d{8,17}\b'),
            'routing_number': re.compile(r'\b\d{9}\b'),
            'aba': re.compile(r'\b\d{9}\b'),
            
            # Government
            'passport': re.compile(r'\b[A-Z0-9]{6,9}\b'),
            'license': re.compile(r'\b[A-Z0-9]{5,12}\b'),
            'tax_id': re.compile(r'\b\d{2}-\d{7}\b'),
        }
    
    def _initialize_keywords(self) -> Dict[str, List[str]]:
        """
        Initialize PII keyword detection - CONSERVATIVE APPROACH
        
        Only flags truly sensitive PERSONAL identifiers that require regulatory protection.
        Generic business terms (department, position, hospital) are NOT PII.
        Ambiguous words (degree, patient, provider) are NOT included to avoid false positives.
        """
        return {
            'personal_info': [
                'first_name', 'last_name', 'full_name', 'fname', 'lname',
                'home_address', 'street_address', 'residential_address', 'mailing_address',
                'phone_number', 'mobile_number', 'cell_phone', 'telephone_number',
                'email_address',
                'ssn', 'social_security_number', 'sin',
                'date_of_birth', 'dob', 'birth_date', 'birthdate',
                'drivers_license', 'driver_license', 'passport_number', 'national_id'
            ],
            'financial_pci': [
                # Payment Card Industry (PCI-DSS)
                'credit_card', 'credit_card_number', 'card_number', 'cc_number', 'cvv', 'cvv2',
                'bank_account', 'bank_account_number', 'routing_number', 'aba_number', 
                'swift_code', 'iban', 'account_number'
            ],
            'financial_regulated': [
                # SOX, GLBA, banking regulations
                'account_balance', 'transaction_amount', 'trade_amount', 'investment_balance',
                'portfolio_value', 'asset_value', 'revenue', 'profit', 'loss',
                'personal_salary', 'individual_income', 'employee_salary', 'compensation',
                'wire_transfer', 'ach_transfer', 'payment_detail'
            ],
            'medical_phi': [
                # Only truly sensitive medical identifiers (PHI under HIPAA)
                'patient_id', 'patient_number', 'mrn', 'medical_record_number',
                'diagnosis_code', 'icd_code', 'procedure_code', 'prescription',
                'health_insurance', 'insurance_id', 'policy_number',
                'lab_result', 'test_result', 'blood_test', 'genetic_test'
            ],
            'education': [
                # Only specific educational PII (student_id, transcript)
                # NOT generic terms like "school" or "degree" (handled by context-aware logic)
                'student_id', 'transcript', 'gpa', 'grade_level'
            ],
            'sensitive_ids': [
                'employee_id', 'emp_id', 'staff_id',
                'customer_id', 'user_id', 'account_id'
            ],
            'biometric': [
                'fingerprint', 'retina_scan', 'iris_scan', 'facial_recognition',
                'biometric_id', 'dna_sequence', 'genetic_profile'
            ]
        }
    
    def _initialize_classifications(self) -> Dict[str, Dict]:
        """Initialize data classification levels"""
        return {
            'PUBLIC': {
                'risk_level': 0,
                'description': 'Public information with no privacy concerns',
                'tags': ['public']
            },
            'INTERNAL': {
                'risk_level': 1,
                'description': 'Internal business information',
                'tags': ['internal', 'business']
            },
            'CONFIDENTIAL': {
                'risk_level': 2,
                'description': 'Confidential information requiring protection',
                'tags': ['confidential', 'protected']
            },
            'RESTRICTED': {
                'risk_level': 3,
                'description': 'Restricted information with legal/regulatory requirements',
                'tags': ['restricted', 'compliance']
            },
            'PII': {
                'risk_level': 4,
                'description': 'Personally Identifiable Information',
                'tags': ['pii', 'personal', 'privacy']
            },
            'PHI': {
                'risk_level': 5,
                'description': 'Protected Health Information (HIPAA)',
                'tags': ['phi', 'health', 'hipaa', 'medical']
            },
            'PCI': {
                'risk_level': 4,
                'description': 'Payment Card Industry data',
                'tags': ['pci', 'financial', 'payment']
            }
        }
    
    def _get_pii_llm_model(self):
        """Get the configured LLM model for PII detection with validation"""
        default_model = 'databricks-gemma-3-12b'
        
        if not self.settings_manager:
            logger.info(f"🤖 No settings manager, using default PII model: {default_model}")
            return default_model
        
        try:
            # Get configured PII model
            pii_config = self.settings_manager.get_pii_config()
            configured_model = pii_config.get('llm_model', default_model)
            
            # Validate that the model is enabled using models config
            try:
                # Import here to avoid circular dependencies
                from models_config import ModelsConfigManager
                models_mgr = ModelsConfigManager(self.settings_manager)
                models = models_mgr.get_available_models()
                
                # Check if model exists and is enabled
                if configured_model not in models or not models[configured_model].get('enabled', False):
                    logger.warning(f"⚠️ PII model '{configured_model}' is disabled. LLM-based detection will be skipped.")
                    return None  # Return None to indicate model is disabled
                
            except Exception as model_check_error:
                logger.warning(f"Could not validate model status: {model_check_error}, assuming enabled")
            
            logger.info(f"🤖 Using PII model: {configured_model}")
            return configured_model
            
        except Exception as e:
            logger.warning(f"Failed to get PII model config: {e}, using default: {default_model}")
            return default_model
    
    def _load_custom_patterns(self):
        """Load custom PII patterns from Settings Manager"""
        try:
            if not self.settings_manager:
                return
                
            # Get PII configuration from settings
            pii_config = self.settings_manager.get_pii_config()
            custom_patterns = pii_config.get('custom_patterns', [])
            
            logger.info(f"🔍 Loading {len(custom_patterns)} custom PII patterns")
            
            # Add custom patterns to keywords
            for pattern in custom_patterns:
                if not pattern.get('enabled', True):
                    continue
                    
                pattern_name = pattern.get('name', '').lower().replace(' ', '_')
                keywords_raw = pattern.get('keywords', '')
                
                # Parse keywords - handle both string and list formats
                if isinstance(keywords_raw, str):
                    # Split comma-separated string and clean up
                    keywords = [k.strip().lower() for k in keywords_raw.split(',') if k.strip()]
                elif isinstance(keywords_raw, list):
                    keywords = [str(k).strip().lower() for k in keywords_raw if str(k).strip()]
                else:
                    keywords = []
                
                if pattern_name and keywords:
                    # Add to custom category
                    if 'custom' not in self.keywords:
                        self.keywords['custom'] = []
                    
                    # Add all keywords for this custom pattern
                    self.keywords['custom'].extend(keywords)
                    
                    logger.info(f"🔍 Added custom PII pattern '{pattern_name}' with keywords: {keywords}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load custom PII patterns: {e}")
    
    def analyze_columns_batch(self, columns: List[Dict], max_batch_size: int = 20) -> List[Dict]:
        """
        Analyze multiple columns for PII with intelligent batch splitting.
        
        Args:
            columns: List of dicts with 'column_name', 'data_type', and 'sample_values'
            max_batch_size: Maximum columns per LLM batch call (default: 20)
        
        Returns:
            List of PII analysis results (same format as analyze_column)
        """
        if not columns:
            return []
        
        logger.info(f"🔍 PII Detection Settings: enabled={self.enabled}, llm_assessment={self.llm_assessment}")
        
        # If we have too many columns, split into multiple batches
        if len(columns) > max_batch_size:
            logger.info(f"🔍 Splitting {len(columns)} columns into batches of {max_batch_size}")
            all_results = []
            for i in range(0, len(columns), max_batch_size):
                batch = columns[i:i+max_batch_size]
                batch_results = self._analyze_columns_batch_internal(batch)
                all_results.extend(batch_results)
            return all_results
        else:
            return self._analyze_columns_batch_internal(columns)
    
    def _analyze_columns_batch_internal(self, columns: List[Dict]) -> List[Dict]:
        """Internal method for single batch PII analysis (no splitting)"""
        if not columns:
            return []
        
        logger.info(f"🔍 Batch PII Detection - Analyzing {len(columns)} columns")
        
        if not self.enabled:
            logger.info("ℹ️ PII detection disabled in settings")
            return [self._empty_result() for _ in columns]
        
        # First pass: Pattern-based detection for all columns (fast)
        results = []
        llm_candidates = []  # Columns that need LLM analysis
        
        for col in columns:
            column_name = col.get('column_name', '') or col.get('name', '')
            data_type = col.get('data_type', '')
            sample_values = col.get('sample_values', [])
            
            # Run pattern and keyword detection using existing methods WITH CONTEXT
            name_analysis = self._analyze_column_name(column_name.lower(), data_type, sample_values)
            data_analysis = self._analyze_sample_data(sample_values) if sample_values else {'pii_types': [], 'confidence': 0.0}
            
            combined_types = list(set(name_analysis['pii_types'] + data_analysis['pii_types']))
            confidence = max(name_analysis['confidence'], data_analysis['confidence'])
            
            if combined_types:
                # High confidence from patterns - skip LLM
                # Data patterns (sample analysis) have higher confidence than name analysis
                has_pattern_match = len(data_analysis['pii_types']) > 0
                classification = self._determine_classification(combined_types)
                result = {
                    'pii_types': combined_types,
                    'confidence': confidence,
                    'classification': classification,
                    'method': 'PATTERN' if has_pattern_match else 'KEYWORD',
                    'column_name': column_name,
                    'data_type': data_type,
                    'proposed_policy_tags': self._generate_proposed_policy_tags(combined_types, classification),
                    'policy_tags': [],  # No automatic tagging
                    'risk_factors': self._assess_risk_factors(combined_types, data_type),
                    'recommendations': []
                }
                results.append(result)
            else:
                # No pattern match - candidate for LLM batch analysis
                llm_candidates.append({
                    'column_name': column_name,
                    'data_type': data_type,
                    'sample_values': sample_values,
                    'index': len(results)  # Track position in results
                })
                # Placeholder result (will be replaced by LLM result)
                results.append(None)
        
        # Second pass: Batch LLM analysis for candidates
        if llm_candidates and self.llm_assessment:
            logger.info(f"🤖 Batch LLM PII Detection for {len(llm_candidates)} columns (single API call)")
            llm_results = self._analyze_batch_with_llm(llm_candidates)
            
            # Insert LLM results back into results list
            for i, llm_result in enumerate(llm_results):
                if i < len(llm_candidates):
                    result_index = llm_candidates[i]['index']
                    results[result_index] = llm_result
        
        # Fill any remaining None placeholders with empty results
        for i in range(len(results)):
            if results[i] is None:
                results[i] = self._empty_result()
        
        logger.info(f"✅ Batch PII Detection complete: {len(results)} columns analyzed")
        return results
    
    def analyze_column(self, column_name: str, data_type: str, sample_values: List[str] = None) -> Dict:
        """
        Analyze a column for PII content and return classification results
        
        Args:
            column_name: Name of the column
            data_type: SQL data type (STRING, INT, etc.)
            sample_values: Optional sample values for analysis
            
        Returns:
            Dict with PII analysis results
        """
        results = {
            'column_name': column_name,
            'data_type': data_type,
            'pii_types': [],
            'confidence_score': 0.0,
            'classification': 'PUBLIC',
            'policy_tags': [],  # Empty - no automatic tagging
            'proposed_policy_tags': [],  # New - proposed tags for manual review
            'risk_factors': [],
            'recommendations': []
        }
        
        # Check if PII detection is enabled in settings
        pii_enabled = True
        llm_assessment_enabled = True
        
        if self.settings_manager:
            try:
                pii_config = self.settings_manager.get_pii_config()
                pii_enabled = pii_config.get('enabled', True)
                llm_assessment_enabled = pii_config.get('llm_assessment_enabled', True)
                
                logger.info(f"🔍 PII Detection Settings: enabled={pii_enabled}, llm_assessment={llm_assessment_enabled}")
                
            except Exception as e:
                logger.error(f"❌ Failed to get PII settings, using defaults: {e}")
        
        # Skip PII detection if disabled in settings
        if not pii_enabled:
            logger.info(f"🔒 PII Detection disabled in settings for column '{column_name}'")
            results['recommendations'].append('PII detection disabled in workspace settings')
            return results
        
        # Analyze column name for PII indicators
        name_analysis = self._analyze_column_name(column_name.lower())
        results['pii_types'].extend(name_analysis['pii_types'])
        results['confidence_score'] = max(results['confidence_score'], name_analysis['confidence'])
        
        # Analyze sample data if provided
        if sample_values:
            data_analysis = self._analyze_sample_data(sample_values)
            results['pii_types'].extend(data_analysis['pii_types'])
            results['confidence_score'] = max(results['confidence_score'], data_analysis['confidence'])
        
        # LLM-based analysis (if enabled and available)
        logger.info(f"🤖 Attempting LLM PII detection for column '{column_name}'")
        llm_analysis = self._analyze_with_llm(
            column_name=column_name,
            data_type=data_type,
            sample_values=sample_values
        )
        
        logger.info(f"🤖 LLM analysis method for '{column_name}': {llm_analysis['method']}")
        
        if llm_analysis['method'] in ['LLM'] and llm_analysis['pii_types']:
            # Merge LLM results with pattern-based results
            results['pii_types'].extend(llm_analysis['pii_types'])
            results['confidence_score'] = max(results['confidence_score'], llm_analysis['confidence'])
            
            # Use LLM classification if it's higher risk than pattern-based
            llm_classification = llm_analysis.get('classification', 'PUBLIC')
            if self._classification_rank(llm_classification) > self._classification_rank(results['classification']):
                results['classification'] = llm_classification
            
            # Add LLM method to recommendations
            results['recommendations'].append(f"LLM detected: {', '.join(llm_analysis['pii_types'])}")
            
            logger.info(f"✅ LLM enhanced PII detection for '{column_name}': {llm_analysis['pii_types']}")
        elif llm_analysis['method'] in ['DISABLED', 'UNAVAILABLE']:
            logger.info(f"ℹ️ LLM PII detection not used for '{column_name}': {llm_analysis['method']}")
        
        # Determine classification and generate PROPOSED tags
        results['classification'] = self._determine_classification(results['pii_types'])
        results['proposed_policy_tags'] = self._generate_proposed_policy_tags(results['pii_types'], results['classification'])
        results['risk_factors'] = self._assess_risk_factors(results['pii_types'], data_type)
        results['recommendations'] = self._generate_recommendations(results)
        
        # Keep old policy_tags field empty (no automatic tagging)
        results['policy_tags'] = []
        
        # Remove duplicates
        results['pii_types'] = list(set(results['pii_types']))
        results['policy_tags'] = list(set(results['policy_tags']))
        
        return results
    
    def _analyze_column_name(self, column_name: str, data_type: str = '', sample_values: List = None) -> Dict:
        """
        Analyze column name for PII indicators with CONTEXT AWARENESS.
        
        Uses data type and sample values to disambiguate ambiguous keywords:
        - "degree" + numeric type + numeric samples = temperature (NOT PII)
        - "degree" + string type = education (PII)
        - "name" with "store_"/"product_" prefix = business name (NOT PII)
        - "name" with "first_"/"last_" prefix = personal name (PII)
        """
        pii_types = []
        confidence = 0.0
        
        # Context-aware ambiguous keyword detection
        # These words can be PII OR non-PII depending on context
        ambiguous_keywords = {
            'degree': {
                'pii_if': lambda dt, sv: dt.upper() in ['STRING', 'VARCHAR', 'TEXT'],  # Education degree
                'not_pii_if': lambda dt, sv: dt.upper() in ['INT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC', 'BIGINT']  # Temperature
            },
            'name': {
                'pii_if': lambda dt, sv: any(prefix in column_name for prefix in ['first_', 'last_', 'full_', 'middle_', 'given_', 'surname']),
                'not_pii_if': lambda dt, sv: any(prefix in column_name for prefix in ['store_', 'product_', 'location_', 'factory_', 'warehouse_', 'business_', 'company_', 'vendor_'])
            },
            'id': {
                'pii_if': lambda dt, sv: any(prefix in column_name for prefix in ['patient_', 'employee_', 'customer_', 'user_', 'person_']),
                'not_pii_if': lambda dt, sv: any(prefix in column_name for prefix in ['product_', 'order_', 'transaction_', 'invoice_', 'shipment_'])
            }
        }
        
        # Check ambiguous keywords with context
        for keyword, rules in ambiguous_keywords.items():
            if keyword in column_name:
                # Check if context indicates NOT PII
                if rules['not_pii_if'](data_type, sample_values):
                    # Skip this keyword - it's not PII in this context
                    continue
                # Check if context indicates PII
                elif rules['pii_if'](data_type, sample_values):
                    pii_types.append(f"personal_info_{keyword}")
                    confidence = max(confidence, 0.8)
        
        # Direct keyword matching for unambiguous terms
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                # Skip already-handled ambiguous keywords
                if keyword in ambiguous_keywords:
                    continue
                    
                if keyword in column_name:
                    pii_types.append(f"{category}_{keyword}")
                    confidence = max(confidence, 0.8)
        
        # Pattern-based detection for truly sensitive formats
        sensitive_patterns = {
            'email': ['email', '@'],
            'phone': ['phone_number', 'mobile_number', 'cell_phone'],  # Must be specific
            'ssn': ['ssn', 'social_security'],
            'personal_address': ['home_address', 'residential_address', 'street_address', 'mailing_address']
        }
        
        for pattern_type, indicators in sensitive_patterns.items():
            if any(indicator in column_name for indicator in indicators):
                pii_types.append(pattern_type)
                confidence = max(confidence, 0.7)
        
        return {
            'pii_types': pii_types,
            'confidence': confidence
        }
    
    def _analyze_sample_data(self, sample_values: List[str]) -> Dict:
        """Analyze sample data for PII patterns"""
        pii_types = []
        confidence = 0.0
        
        if not sample_values:
            return {'pii_types': [], 'confidence': 0.0}
        
        # Convert to strings and clean
        clean_samples = []
        for value in sample_values[:10]:  # Analyze first 10 samples
            if value is not None:
                clean_samples.append(str(value).strip())
        
        if not clean_samples:
            return {'pii_types': [], 'confidence': 0.0}
        
        # Pattern matching on sample data
        for pattern_name, pattern in self.patterns.items():
            matches = 0
            for sample in clean_samples:
                if pattern.search(sample):
                    matches += 1
            
            if matches > 0:
                match_ratio = matches / len(clean_samples)
                pii_types.append(pattern_name)
                confidence = max(confidence, match_ratio * 0.9)
        
        return {
            'pii_types': pii_types,
            'confidence': confidence
        }
    
    def _determine_classification(self, pii_types: List[str]) -> str:
        """
        Determine data classification based on detected PII types.
        
        Uses conservative logic to avoid over-classification.
        Only truly sensitive data gets PHI/PCI/PII classification.
        """
        if not pii_types:
            return 'PUBLIC'
        
        pii_str = ''.join(pii_types).lower()
        
        # PHI: Protected Health Information (HIPAA-regulated)
        phi_indicators = ['medical_phi', 'patient_id', 'mrn', 'diagnosis', 'prescription', 'health_insurance']
        if any(indicator in pii_str for indicator in phi_indicators):
            return 'PHI'
        
        # PCI: Payment Card Industry (credit card data) - PCI-DSS compliance
        pci_indicators = ['financial_pci', 'credit_card', 'cvv', 'cvv2', 'card_number', 'bank_account', 'routing']
        if any(indicator in pii_str for indicator in pci_indicators):
            return 'PCI'
        
        # FINANCIAL: Regulated financial data (SOX, GLBA) - not PCI but still sensitive
        financial_indicators = ['financial_regulated', 'account_balance', 'transaction_amount', 
                               'trade_amount', 'portfolio', 'investment', 'wire_transfer', 'ach_transfer']
        if any(indicator in pii_str for indicator in financial_indicators):
            return 'CONFIDENTIAL'  # Financial data is confidential but not PCI
        
        # PII: Personally Identifiable Information (SSN, address, biometric, education records)
        pii_indicators = ['personal_info', 'ssn', 'social_security', 'home_address', 
                         'passport', 'drivers_license', 'biometric', 'dna', 'education', 
                         'student_id', 'transcript', 'gpa']
        if any(indicator in pii_str for indicator in pii_indicators):
            return 'PII'
        
        # Sensitive IDs: Internal identifiers (employee_id, customer_id)
        # These are INTERNAL, not PII - they don't expose personal information by themselves
        id_indicators = ['sensitive_ids', 'employee_id', 'customer_id', 'user_id']
        if any(indicator in pii_str for indicator in id_indicators):
            return 'INTERNAL'
        
        # Default to INTERNAL for any other detected types
        return 'INTERNAL' if pii_types else 'PUBLIC'
    
    def _generate_proposed_policy_tags(self, pii_types: List[str], classification: str) -> List[Dict]:
        """Generate PROPOSED policy tags based on PII analysis for manual review"""
        proposed_tags = []
        
        # Add classification tag proposal
        if classification != 'PUBLIC':
            proposed_tags.append({
                'tag': f'classification.{classification}',
                'reason': f'Data classified as {classification} based on content analysis',
                'confidence': 'high',
                'auto_apply': False  # Always requires manual approval
            })
        
        # Add specific PII tag proposals
        tag_mapping = {
            'ssn': {'tag': 'PII.SSN', 'reason': 'Social Security Number patterns detected'},
            'email': {'tag': 'PII.Email', 'reason': 'Email address patterns detected'},
            'phone': {'tag': 'PII.Phone', 'reason': 'Phone number patterns detected'},
            'credit_card': {'tag': 'PCI.CreditCard', 'reason': 'Credit card number patterns detected'},
            'medical': {'tag': 'PHI.Medical', 'reason': 'Medical record patterns detected'},
            'patient': {'tag': 'PHI.Patient', 'reason': 'Patient information patterns detected'},
            'financial': {'tag': 'PCI.Financial', 'reason': 'Financial data patterns detected'},
            'personal_info': {'tag': 'PII.Personal', 'reason': 'Personal information patterns detected'}
        }
        
        for pii_type in pii_types:
            for key, tag_info in tag_mapping.items():
                if key in pii_type.lower():
                    proposed_tags.append({
                        'tag': tag_info['tag'],
                        'reason': tag_info['reason'],
                        'confidence': 'high' if key in ['ssn', 'credit_card', 'email'] else 'medium',
                        'auto_apply': False,  # Always requires manual approval
                        'pii_type': pii_type
                    })
        
        return proposed_tags
    
    def _assess_risk_factors(self, pii_types: List[str], data_type: str) -> List[str]:
        """Assess risk factors for the detected PII"""
        risks = []
        
        if pii_types:
            risks.append('Contains personally identifiable information')
        
        if 'medical' in ''.join(pii_types).lower():
            risks.append('Subject to HIPAA regulations')
        
        if 'financial' in ''.join(pii_types).lower():
            risks.append('Subject to PCI DSS requirements')
        
        if data_type == 'STRING' and any('id' in pii for pii in pii_types):
            risks.append('Potential unique identifier')
        
        return risks
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on PII analysis"""
        recommendations = []
        
        classification = analysis.get('classification', 'PUBLIC')
        pii_types = analysis.get('pii_types', [])
        
        if classification in ['PII', 'PHI', 'PCI']:
            recommendations.append('Consider data masking or tokenization')
            recommendations.append('Implement access controls and audit logging')
            recommendations.append('Ensure compliance with relevant regulations')
        
        if 'credit_card' in ''.join(pii_types).lower():
            recommendations.append('Apply PCI DSS security controls')
        
        if 'medical' in ''.join(pii_types).lower():
            recommendations.append('Implement HIPAA safeguards')
        
        if analysis.get('confidence_score', 0) < 0.5:
            recommendations.append('Manual review recommended - low confidence detection')
        
        return recommendations

    def analyze_table(self, table_info: Dict) -> Dict:
        """
        Analyze entire table for PII content
        
        Args:
            table_info: Dictionary with table metadata and sample data
            
        Returns:
            Dict with table-level PII analysis
        """
        table_name = table_info.get('name', '')
        columns = table_info.get('columns', [])
        
        results = {
            'table_name': table_name,
            'total_columns': len(columns),
            'pii_columns': 0,
            'highest_classification': 'PUBLIC',
            'recommended_tags': [],
            'risk_assessment': 'LOW',
            'column_analysis': []
        }
        
        # Use BATCH analysis for all columns at once (100x faster than one-by-one!)
        logger.info(f"🚀 Analyzing table '{table_name}' with {len(columns)} columns using BATCH PII detection")
        column_analyses = self.analyze_columns_batch(columns, max_batch_size=20)
        
        # Aggregate results
        for column_analysis in column_analyses:
            results['column_analysis'].append(column_analysis)
            
            # Track PII columns
            if column_analysis['classification'] != 'PUBLIC':
                results['pii_columns'] += 1
            
            # Update highest classification
            if self._classification_rank(column_analysis['classification']) > self._classification_rank(results['highest_classification']):
                results['highest_classification'] = column_analysis['classification']
            
            # Collect unique tags
            results['recommended_tags'].extend(column_analysis.get('policy_tags', []))
        
        # Remove duplicate tags
        results['recommended_tags'] = list(set(results['recommended_tags']))
        
        # Determine risk assessment
        pii_ratio = results['pii_columns'] / results['total_columns'] if results['total_columns'] > 0 else 0
        if pii_ratio > 0.5 or results['highest_classification'] in ['PHI', 'PCI']:
            results['risk_assessment'] = 'HIGH'
        elif pii_ratio > 0.2 or results['highest_classification'] == 'PII':
            results['risk_assessment'] = 'MEDIUM'
        else:
            results['risk_assessment'] = 'LOW'
        
        return results
    
    def _analyze_with_llm(self, column_name: str, data_type: str, sample_values: List[str] = None, table_context: Dict = None) -> Dict:
        """
        Use LLM to analyze column for REGULATED/SENSITIVE DATA across ALL frameworks.
        
        Detects:
        - PII: Personal information (GDPR, CCPA)
        - PHI: Protected Health Information (HIPAA)
        - PCI: Payment Card data (PCI-DSS)
        - Financial: Banking/trading data (SOX, GLBA)
        - Other regulated data requiring compliance controls
        """
        try:
            if not self.llm_service:
                logger.warning("🤖 LLM service not available for regulatory data detection")
                return {'pii_types': [], 'confidence': 0.0, 'method': 'UNAVAILABLE'}
            
            # Check if LLM detection is enabled
            llm_enabled = True
            if self.settings_manager:
                try:
                    pii_config = self.settings_manager.get_pii_config()
                    llm_enabled = pii_config.get('llm_detection_enabled', True)
                except Exception as e:
                    logger.warning(f"Failed to check LLM settings: {e}")
            
            if not llm_enabled:
                logger.info(f"🤖 LLM regulatory data detection disabled in settings for column '{column_name}'")
                return {'pii_types': [], 'confidence': 0.0, 'method': 'DISABLED'}
            
            # Build context for LLM
            table_info = ""
            if table_context:
                table_info = f"Table: {table_context.get('schema_name', '')}.{table_context.get('table_name', '')}\n"
            
            sample_info = ""
            if sample_values and len(sample_values) > 0:
                # Show first few sample values (anonymized)
                samples = sample_values[:3]
                sample_info = f"Sample values: {', '.join(str(s)[:20] + '...' if len(str(s)) > 20 else str(s) for s in samples)}\n"
            
            prompt = f"""Analyze this database column for REGULATED/SENSITIVE DATA requiring compliance controls:

Column Name: {column_name}
Data Type: {data_type}
{table_info}{sample_info}

REGULATORY FRAMEWORKS TO CONSIDER:
1. PII (GDPR/CCPA): Personal identifiers (name, email, SSN, address, phone, DOB, passport, license)
2. PHI (HIPAA): Health information (patient records, diagnoses, prescriptions, medical IDs, insurance)
3. PCI (PCI-DSS): Payment data (credit cards, bank accounts, CVV, routing numbers, transaction details)
4. FINANCIAL (SOX/GLBA): Banking/trading data (account numbers, balances, trades, investment records)
5. BIOMETRIC: Fingerprints, facial recognition, DNA, retina scans
6. CONFIDENTIAL: Trade secrets, proprietary algorithms, competitive intelligence

Consider:
- Column name patterns and business context
- Data type appropriateness for sensitive data
- Sample values and data patterns
- Regulatory exposure if this data were breached

Respond with ONLY this format:
PII_TYPES: [comma-separated list like "email", "ssn", "credit_card", "patient_id", "diagnosis", "account_balance" or "none"]
CONFIDENCE: [number 0.0-1.0]
CLASSIFICATION: [PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED, PII, PHI, PCI]"""

            logger.info(f"🔒 LLM Regulatory Data Detection - Analyzing column '{column_name}' (type: {data_type})")
            logger.info(f"🔒 LLM Prompt (first 200 chars):\n{prompt[:200]}...")  # Log first 200 chars of prompt
            
            # Refresh model setting to ensure we use the latest config
            self.llm_model = self._get_pii_llm_model()
            
            # Skip LLM detection if model is disabled
            if not self.llm_model:
                logger.info(f"⏭️  Skipping LLM PII detection for {column_name} - model is disabled")
                return result
            
            response = self.llm_service._call_databricks_llm(
                prompt=prompt,
                max_tokens=100,
                model=self.llm_model,
                temperature=0.1,
                style="concise"
            )
            
            if not response:
                logger.warning(f"🤖 Empty LLM response for column '{column_name}'")
                return {'pii_types': [], 'confidence': 0.0, 'method': 'FAILED'}
            
            # Parse LLM response
            pii_types = []
            confidence = 0.0
            classification = 'PUBLIC'
            
            import re
            
            # Extract PII types
            pii_match = re.search(r'PII_TYPES:\s*\[(.*?)\]', response, re.IGNORECASE)
            if pii_match:
                pii_text = pii_match.group(1).strip()
                if pii_text.lower() != 'none' and pii_text:
                    pii_types = [t.strip().strip('"\'') for t in pii_text.split(',') if t.strip()]
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
            if conf_match:
                confidence = min(1.0, max(0.0, float(conf_match.group(1))))
            
            # Extract classification
            class_match = re.search(r'CLASSIFICATION:\s*(\w+)', response, re.IGNORECASE)
            if class_match:
                classification = class_match.group(1).upper()
            
            logger.info(f"🤖 LLM PII Detection Result for '{column_name}':")
            logger.info(f"   - PII Types: {pii_types if pii_types else 'None'}")
            logger.info(f"   - Confidence: {confidence:.2f}")
            logger.info(f"   - Classification: {classification}")
            logger.info(f"   - Raw Response: {response[:150]}...")  # Log first 150 chars
            
            return {
                'pii_types': pii_types,
                'confidence': confidence,
                'classification': classification,
                'method': 'LLM',
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"🤖 LLM PII detection failed for column '{column_name}': {e}")
            return {'pii_types': [], 'confidence': 0.0, 'method': 'ERROR'}
    
    def _analyze_batch_with_llm(self, columns: List[Dict]) -> List[Dict]:
        """
        Analyze multiple columns with LLM for REGULATED/SENSITIVE DATA in a single batch call.
        
        Detects ALL regulatory frameworks: PII, PHI, PCI, Financial, Biometric, Confidential.
        Much more efficient than one-by-one analysis.
        """
        if not columns:
            return []
        
        try:
            if not self.llm_service:
                logger.warning("🤖 LLM service not available for batch regulatory data detection")
                return [{'pii_types': [], 'confidence': 0.0, 'method': 'UNAVAILABLE', 'proposed_policy_tags': []} for _ in columns]
            
            # Check if LLM detection is enabled
            llm_enabled = True
            if self.settings_manager:
                try:
                    pii_config = self.settings_manager.get_pii_config()
                    llm_enabled = pii_config.get('llm_detection_enabled', True)
                except Exception as e:
                    logger.warning(f"Failed to check LLM settings: {e}")
            
            if not llm_enabled:
                logger.info(f"🤖 LLM regulatory data detection disabled in settings")
                return [{'pii_types': [], 'confidence': 0.0, 'method': 'DISABLED', 'proposed_policy_tags': []} for _ in columns]
            
            # Build batch prompt
            prompt = f"""Analyze these {len(columns)} database columns for REGULATED/SENSITIVE DATA requiring compliance controls:

"""
            
            for i, col in enumerate(columns, 1):
                column_name = col['column_name']
                data_type = col['data_type']
                sample_values = col.get('sample_values', [])
                
                sample_info = ""
                if sample_values:
                    samples = sample_values[:3]
                    sample_str = ', '.join(str(s)[:20] + '...' if len(str(s)) > 20 else str(s) for s in samples)
                    sample_info = f" | Sample values: {sample_str}"
                
                prompt += f"{i}. {column_name} ({data_type}){sample_info}\n"
            
            prompt += f"""
REGULATORY FRAMEWORKS:
- PII (GDPR/CCPA): Personal identifiers (name, email, SSN, phone, address, DOB)
- PHI (HIPAA): Health data (diagnoses, prescriptions, patient IDs, medical records)
- PCI (PCI-DSS): Payment data (credit cards, bank accounts, CVV, transactions)
- FINANCIAL (SOX/GLBA): Banking/trading data (balances, account numbers, trades)
- BIOMETRIC: Fingerprints, facial recognition, DNA
- CONFIDENTIAL: Trade secrets, proprietary data

For EACH column, classify based on regulatory exposure if breached. Consider:
- Column name patterns and business context
- Sample values and data patterns
- Data type appropriateness for sensitive data

Respond with EXACTLY {len(columns)} lines in this format (one per column):
PII_TYPES: [types like "email", "ssn", "credit_card", "diagnosis", "account_balance" or "none"] | CONFIDENCE: [0.0-1.0] | CLASSIFICATION: [PUBLIC/INTERNAL/PII/PHI/PCI]
"""
            
            logger.info(f"🔒 Batch LLM Regulatory Detection Prompt (first 300 chars):\n{prompt[:300]}...")
            logger.info(f"Calling LLM with prompt length: {len(prompt)}")
            
            # Refresh model setting to ensure we use the latest config
            self.llm_model = self._get_pii_llm_model()
            
            # Call LLM
            # Skip LLM detection if model is disabled
            if not self.llm_model:
                logger.info(f"⏭️  Skipping batch LLM PII detection for {len(columns)} columns - model is disabled")
                return [{'column_name': col['name'], 'pii_types': [], 'confidence': 0.0, 'method': 'SKIPPED'} for col in columns]
            
            response = self.llm_service._call_databricks_llm(
                prompt=prompt,
                max_tokens=1000,  # More tokens for batch response
                model=self.llm_model,
                temperature=0.1,
                style="concise"
            )
            
            if not response:
                logger.warning("❌ Empty response from LLM for batch PII detection")
                return [self._empty_result() for _ in columns]
            
            logger.info(f"🤖 Batch LLM Response (first 300 chars):\n{str(response)[:300]}...")
            
            # Parse batch response
            results = self._parse_batch_llm_response(response, columns)
            
            logger.info(f"✅ Batch LLM PII Detection complete: {len(results)} columns analyzed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch LLM PII detection error: {e}")
            return [self._empty_result() for _ in columns]
    
    def _parse_batch_llm_response(self, response: Any, columns: List[Dict]) -> List[Dict]:
        """Parse LLM batch response into individual column results"""
        try:
            # Extract text content from response
            response_text = ""
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
                return [self._empty_result() for _ in columns]
            
            # Split into lines
            lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
            
            results = []
            for i, col in enumerate(columns):
                if i < len(lines):
                    line = lines[i]
                    result = self._parse_single_pii_response(line)
                    result['column_name'] = col['column_name']
                    result['data_type'] = col.get('data_type', '')
                    result['method'] = 'LLM'
                    result['proposed_policy_tags'] = self._generate_proposed_policy_tags(result['pii_types'], result['classification'])
                    result['policy_tags'] = []  # No automatic tagging
                    result['risk_factors'] = self._assess_risk_factors(result['pii_types'], result['data_type'])
                    result['recommendations'] = []
                    results.append(result)
                else:
                    # Not enough lines in response
                    results.append(self._empty_result())
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to parse batch LLM response: {e}")
            return [self._empty_result() for _ in columns]
    
    def _parse_single_pii_response(self, response_line: str) -> Dict:
        """Parse a single line of PII response"""
        try:
            import re
            pii_types = []
            confidence = 0.0
            classification = 'PUBLIC'
            
            # Extract PII_TYPES
            if 'PII_TYPES:' in response_line:
                types_part = response_line.split('PII_TYPES:')[1].split('|')[0].strip()
                types_part = types_part.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                if 'none' not in types_part.lower():
                    pii_types = [t.strip() for t in types_part.split(',') if t.strip()]
            
            # Extract CONFIDENCE
            if 'CONFIDENCE:' in response_line:
                conf_part = response_line.split('CONFIDENCE:')[1].split('|')[0].strip()
                conf_part = conf_part.replace('[', '').replace(']', '')
                try:
                    confidence = float(conf_part)
                except:
                    confidence = 0.8 if pii_types else 0.0
            
            # Extract CLASSIFICATION
            if 'CLASSIFICATION:' in response_line:
                class_part = response_line.split('CLASSIFICATION:')[1].strip()
                class_part = class_part.replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
                if class_part in ['PUBLIC', 'PII', 'PHI', 'PCI']:
                    classification = class_part
            
            return {
                'pii_types': pii_types,
                'confidence': confidence,
                'classification': classification
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse PII response line: {e}")
            return {'pii_types': [], 'confidence': 0.0, 'classification': 'PUBLIC'}
    
    def _empty_result(self) -> Dict:
        """Return empty PII analysis result"""
        return {
            'pii_types': [],
            'confidence': 0.0,
            'classification': 'PUBLIC',
            'method': 'NONE',
            'proposed_policy_tags': []
        }
    
    def _classification_rank(self, classification: str) -> int:
        """Get numeric rank for classification level"""
        rankings = {
            'PUBLIC': 0,
            'INTERNAL': 1,
            'CONFIDENTIAL': 2,
            'RESTRICTED': 3,
            'PII': 4,
            'PCI': 4,
            'PHI': 5
        }
        return rankings.get(classification, 0)
