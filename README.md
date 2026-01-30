# Unity Catalog Metadata Assistant

A production-ready Databricks app for AI-powered metadata generation, PII detection, and governance analytics. Fully self-contained with zero external dependencies.

---

## 🚀 Setup from Scratch

Follow these steps to deploy the UC Metadata Assistant in your Databricks workspace.

### Prerequisites

- Databricks Workspace with Unity Catalog enabled
- SQL Warehouse (Pro or Serverless recommended)
- At least one LLM Serving Endpoint (Databricks Foundation Models)
- Workspace Admin or sufficient permissions to create apps

### Step 1: Prepare SQL Warehouse

1. Go to **SQL Warehouses** in your Databricks workspace
2. Create or select an existing SQL Warehouse
3. Note the **HTTP Path**: `/sql/1.0/warehouses/<warehouse-id>`
4. Note your **Workspace Hostname**: `<your-workspace>.cloud.databricks.com`

### Step 2: Create the App Catalog

The app requires a dedicated catalog for storing settings and metadata. Creating it manually avoids needing to grant `CREATE CATALOG` permission to the app service principal.

```sql
-- Create the app catalog (if it doesn't exist)
CREATE CATALOG IF NOT EXISTS uc_metadata_assistant;

-- Verify creation
SHOW CATALOGS LIKE 'uc_metadata_assistant';
```

**Why create manually?**
- Avoids granting broad `CREATE CATALOG ON METASTORE` permission
- Follows security best practice (least privilege)
- Gives you control over catalog location and properties

The app will automatically create the required schemas and tables on first launch.

### Step 3: Clone Repository to Workspace

Use Databricks Git integration:

1. Go to **Workspace** → **Repos** in your Databricks workspace
2. Click **Add Repo**
3. Enter your Git repository URL
4. Click **Create Repo**

Or use Databricks CLI:

```bash
# Clone to workspace folder
databricks repos create \
  --url <your-git-repo-url> \
  --provider github \
  --path /Repos/<your-username>/governance_app
```

### Step 4: Configure app.yaml

Edit `app.yaml` in the cloned repository:

```yaml
command: [
  "flask",
  "--app", 
  "app_react:flask_app",
  "run",
  "--host=0.0.0.0"
]

env:
  - name: 'DATABRICKS_HTTP_PATH'
    value: '<Your Warehouse HTTP Path>' # ← Update this
  - name: 'DATABRICKS_SERVER_HOSTNAME'
    value: '<Your Server Hostname>' # ← Update this
```

**Note**: `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `DATABRICKS_HOST`, and `DATABRICKS_WORKSPACE_ID` are automatically provided by Databricks Apps.

### Step 5: Deploy the App

Using Databricks CLI:

```bash
# Navigate to the repository folder
cd /path/to/governance_app

# Deploy the app
databricks apps deploy uc-metadata-assistant \
  --source-code-path /Repos/<your-username>/governance_app
```

Or use the Databricks Apps UI:

1. Go to **Apps** in your workspace
2. Click **Create App**
3. Name: `uc-metadata-assistant`
4. Source: Select the repo folder path
5. Click **Create**

### Step 6: Grant Permissions to App Service Principal

After deployment, the app creates a service principal. Grant it the required permissions:

#### 6a. Find the App Service Principal

```bash
# List apps and get service principal
databricks apps list

# Note the service principal name (usually: app-<app-name>-<id>)
```

Or find it in the Apps UI → Your App → Configuration tab.

#### 6b. Grant Required Permissions

Replace `<app-service-principal>` with your app's service principal name:

```sql
-- ============================================
-- CORE PERMISSIONS (Required for all features)
-- ============================================

-- App Infrastructure (catalog for settings/cache)
-- Note: CREATE CATALOG not needed if you created uc_metadata_assistant in Step 2
GRANT CREATE SCHEMA ON CATALOG uc_metadata_assistant TO `<app-service-principal>`;
GRANT ALL PRIVILEGES ON CATALOG uc_metadata_assistant TO `<app-service-principal>`;

-- ============================================
-- METADATA DISCOVERY (Required)
-- ============================================

-- For each catalog you want to document:
GRANT USE CATALOG ON CATALOG <target-catalog> TO `<app-service-principal>`;
GRANT USE SCHEMA ON SCHEMA <target-catalog>.* TO `<app-service-principal>`;
GRANT SELECT ON SCHEMA <target-catalog>.* TO `<app-service-principal>`;
GRANT SELECT ON TABLE <target-catalog>.*.* TO `<app-service-principal>`;

-- ============================================
-- METADATA GENERATION (Required for generating descriptions)
-- ============================================

-- MODIFY permission required for COMMENT statements
GRANT MODIFY ON CATALOG <target-catalog> TO `<app-service-principal>`;
GRANT MODIFY ON SCHEMA <target-catalog>.* TO `<app-service-principal>`;
GRANT MODIFY ON TABLE <target-catalog>.*.* TO `<app-service-principal>`;

-- ============================================
-- TAG MANAGEMENT (Required for PII detection and tag import)
-- ============================================

-- Apply tags to schemas
GRANT APPLY TAG ON CATALOG <target-catalog> TO `<app-service-principal>`;
GRANT APPLY TAG ON SCHEMA <target-catalog>.* TO `<app-service-principal>`;
GRANT APPLY TAG ON TABLE <target-catalog>.*.* TO `<app-service-principal>`;

-- Read governed tag definitions
GRANT USE CATALOG ON CATALOG system TO `<app-service-principal>`;
GRANT USE SCHEMA ON SCHEMA system.information_schema TO `<app-service-principal>`;
GRANT SELECT ON SCHEMA system.information_schema TO `<app-service-principal>`;

-- ============================================
-- DATA SAMPLING (Optional - for better AI descriptions)
-- ============================================

-- Allows the app to read sample rows from tables
-- (Already granted via SELECT TABLE permissions above)

-- ============================================
-- LLM MODEL ACCESS (Required for AI generation)
-- ============================================

-- Grant access to Foundation Model serving endpoints
GRANT EXECUTE ON ANY FILE TO `<app-service-principal>`;  -- For model serving access

-- Or grant per-model:
-- GRANT EXECUTE ON MODEL <model-endpoint> TO `<app-service-principal>`;

-- ============================================
-- SQL WAREHOUSE ACCESS (Required)
-- ============================================

-- Grant warehouse usage
GRANT USE ON SQL WAREHOUSE <warehouse-id> TO `<app-service-principal>`;

-- ============================================
-- OPTIONAL: Auto-create catalog (not recommended)
-- ============================================

-- Only needed if you skip Step 2 and want the app to auto-create uc_metadata_assistant
-- NOT RECOMMENDED: Grants broad metastore-level permission
-- GRANT CREATE CATALOG ON METASTORE TO `<app-service-principal>`;
```

#### 6c. Quick Permission Grant Script

For convenience, here's a complete script to grant all permissions for a single catalog:

```sql
-- Replace these variables
SET VAR catalog_name = 'your_catalog_name';
SET VAR app_sp = 'app-uc-metadata-assistant-xxxxx';
SET VAR warehouse_id = 'your_warehouse_id';

-- App infrastructure (assumes uc_metadata_assistant catalog exists from Step 2)
GRANT ALL PRIVILEGES ON CATALOG uc_metadata_assistant TO IDENTIFIER($app_sp);

-- Target catalog permissions
GRANT USE CATALOG ON CATALOG IDENTIFIER($catalog_name) TO IDENTIFIER($app_sp);
GRANT USE SCHEMA ON SCHEMA IDENTIFIER($catalog_name).* TO IDENTIFIER($app_sp);
GRANT SELECT ON SCHEMA IDENTIFIER($catalog_name).* TO IDENTIFIER($app_sp);
GRANT SELECT ON TABLE IDENTIFIER($catalog_name).*.* TO IDENTIFIER($app_sp);
GRANT MODIFY ON CATALOG IDENTIFIER($catalog_name) TO IDENTIFIER($app_sp);
GRANT MODIFY ON SCHEMA IDENTIFIER($catalog_name).* TO IDENTIFIER($app_sp);
GRANT MODIFY ON TABLE IDENTIFIER($catalog_name).*.* TO IDENTIFIER($app_sp);
GRANT APPLY TAG ON CATALOG IDENTIFIER($catalog_name) TO IDENTIFIER($app_sp);
GRANT APPLY TAG ON SCHEMA IDENTIFIER($catalog_name).* TO IDENTIFIER($app_sp);
GRANT APPLY TAG ON TABLE IDENTIFIER($catalog_name).*.* TO IDENTIFIER($app_sp);

-- System catalog for governed tags
GRANT USE CATALOG ON CATALOG system TO IDENTIFIER($app_sp);
GRANT USE SCHEMA ON SCHEMA system.information_schema TO IDENTIFIER($app_sp);
GRANT SELECT ON SCHEMA system.information_schema TO IDENTIFIER($app_sp);

-- SQL Warehouse
GRANT USE ON SQL WAREHOUSE IDENTIFIER($warehouse_id) TO IDENTIFIER($app_sp);

-- Model serving access
GRANT EXECUTE ON ANY FILE TO IDENTIFIER($app_sp);
```

### Step 7: Access the App

1. Go to **Apps** in your Databricks workspace
2. Click on **uc-metadata-assistant**
3. Click **Open App** or use the provided URL
4. The app will automatically:
   - Create required schemas and tables in `uc_metadata_assistant`
   - Discover available catalogs
   - Load default models from Databricks Foundation Models

### Step 8: Configure Models (Optional)

1. Navigate to **Settings** → **Models** tab
2. Enable/disable desired models (Claude, Llama, Gemma, GPT, etc.)
3. Add custom model endpoints from your Databricks Serving page
4. Configure max tokens and other parameters

🎉 **You're ready to start generating metadata!**

---

## Setup Checklist

Use this checklist to verify your deployment:

- [ ] **SQL Warehouse**: Created/identified and running
- [ ] **App Catalog**: `uc_metadata_assistant` catalog created manually (Step 2)
- [ ] **Repository**: Code cloned to Databricks Workspace/Repos
- [ ] **Configuration**: `app.yaml` updated with warehouse HTTP path and hostname
- [ ] **Deployment**: App deployed via CLI or UI
- [ ] **Service Principal**: App service principal identified
- [ ] **Core Permissions**: `ALL PRIVILEGES` granted on `uc_metadata_assistant` catalog
- [ ] **Discovery Permissions**: `USE CATALOG`, `USE SCHEMA`, `SELECT` granted on target catalogs
- [ ] **Write Permissions**: `MODIFY` granted for description generation
- [ ] **Tag Permissions**: `APPLY TAG` granted for PII and import features
- [ ] **System Access**: `SELECT system.information_schema` granted for governed tags
- [ ] **Warehouse Access**: `USE SQL WAREHOUSE` granted
- [ ] **Model Access**: `EXECUTE ON ANY FILE` granted for model serving
- [ ] **App Accessible**: App URL loads successfully
- [ ] **Models Configured**: At least one LLM model enabled in Settings

**Note**: `CREATE CATALOG ON METASTORE` is not needed since you created the catalog in Step 2.

**Minimal Working Setup** (for testing):
```sql
-- Replace variables
SET VAR app_sp = 'app-uc-metadata-assistant-xxxxx';
SET VAR catalog = 'your_test_catalog';
SET VAR warehouse = 'your_warehouse_id';

-- Essential permissions only
GRANT ALL PRIVILEGES ON CATALOG uc_metadata_assistant TO IDENTIFIER($app_sp);
GRANT USE CATALOG ON CATALOG IDENTIFIER($catalog) TO IDENTIFIER($app_sp);
GRANT USE SCHEMA ON SCHEMA IDENTIFIER($catalog).* TO IDENTIFIER($app_sp);
GRANT SELECT ON TABLE IDENTIFIER($catalog).*.* TO IDENTIFIER($app_sp);
GRANT MODIFY ON CATALOG IDENTIFIER($catalog) TO IDENTIFIER($app_sp);
GRANT USE ON SQL WAREHOUSE IDENTIFIER($warehouse) TO IDENTIFIER($app_sp);
GRANT EXECUTE ON ANY FILE TO IDENTIFIER($app_sp);
```

---

## Permission Requirements by Feature

| Feature | Required Permissions | Purpose |
|---------|---------------------|---------|
| **App Infrastructure** | `ALL PRIVILEGES` on `uc_metadata_assistant` | Store settings, cache, audit logs |
| **Metadata Discovery** | `USE CATALOG`, `USE SCHEMA`, `SELECT TABLE` | Browse catalogs, schemas, tables, columns |
| **Description Generation** | `MODIFY CATALOG/SCHEMA/TABLE` | Write COMMENT statements to Unity Catalog |
| **PII Detection** | `SELECT TABLE` (for sampling), `APPLY TAG` (for tags) | Read sample data, apply PII tags |
| **Tag Import** | `APPLY TAG` | Import tags from CSV files |
| **Copy Metadata** | `SELECT` (source), `MODIFY` (target) | Copy descriptions between objects |
| **Quality Metrics** | `SELECT TABLE` | Calculate completeness and coverage |
| **Governed Tag Validation** | `SELECT system.information_schema` | Read tag policies and allowed values |
| **Data Sampling** | `SELECT TABLE` | Read sample rows for AI context |
| **LLM Model Access** | `EXECUTE ON ANY FILE` | Access Databricks Foundation Model endpoints |
| **SQL Warehouse** | `USE SQL WAREHOUSE` | Execute queries and read metadata |

**Note**: `CREATE CATALOG ON METASTORE` is **not required** if you create the `uc_metadata_assistant` catalog manually (recommended in Step 2).

---

## Versions

This repository includes two application versions:

- **React Version (Recommended)**: `app_react.py` + `app.yaml` - Modern React frontend with enhanced UI/UX
- **Legacy Version**: `app.py` + `app.yaml` - Original Flask application with embedded frontend

**Important**: The React version depends on the original `app.py` file for all backend functionality. Both files must be present to run the React version.

---

## Quick Reference

**Already set up?** Jump to:
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

**New to the app?** Follow the [Setup from Scratch](#-setup-from-scratch) guide above.

## Features

### AI-Powered Metadata Generation
- Multi-model support (GPT, Llama, Gemma, Claude, custom endpoints)
- Configurable generation styles (concise, technical, business)
- Intelligent context analysis using schema relationships and sample data
- Parallel batch processing for performance
- Real-time progress tracking with ETA and phase indicators
- Dual engine support: UC Assistant (built-in) or dbxmetagen (industry solution)

### Metadata Copy & Sync
- Smart matching with pattern detection (bronze→silver→gold)
- Two-column UI with confidence scoring
- Manual mapping for edge cases
- Bulk operations with batch processing
- Export to CSV for backup or cross-workspace sync
- Import from CSV with validation and preview
- Dev→Test→Prod promotion workflows

### Enterprise PII Detection
- 50+ built-in detection patterns across 8 compliance frameworks
- Pattern-based detection (column names, data types)
- LLM-enhanced detection for context analysis and data sampling
- Configurable tag mappings for governed tag integration
- Redaction support for sensitive data in prompts

### Quality & Governance
- Real-time completeness, accuracy, and tag coverage metrics
- Historical trend analysis with 90-day completeness trends
- Audit trails for all metadata operations
- Governed tag policy enforcement
- AI-generated content quality scoring

### Advanced Settings
- Model management (enable/disable, add custom endpoints)
- PII pattern configuration (50+ patterns, add custom)
- Tag policy management (governed tags, manual tags)
- Data sampling controls with intelligent batching
- Prompt customization (length, focus, terminology, custom instructions)
- Generation mode toggle (UC Assistant vs. dbxmetagen)

## Architecture

```
governance_app/
├── app_react.py              # React version - Main Flask backend (RECOMMENDED)
├── app.yaml            # React version - Databricks app configuration
├── app.py                    # Legacy version - Original Flask app
├── app.yaml                  # Legacy version - App configuration
├── enhanced_generator.py     # AI metadata generation engine (shared)
├── pii_detector.py           # PII/PHI detection engine (shared)
├── metadata_copy_utils.py    # Metadata copy & sync utilities
├── dbxmetagen_adapter.py     # dbxmetagen integration adapter
├── setup_utils.py            # Auto infrastructure setup (shared)
├── settings_manager.py       # Settings persistence (shared)
├── models_config.py          # LLM model management (shared)
├── requirements.txt          # Python dependencies (shared)
└── client/build/             # Pre-built React frontend (React version only)
    ├── index.html
    └── assets/
```

## Automatic Infrastructure

The app automatically creates:

```
uc_metadata_assistant.generated_metadata.*
uc_metadata_assistant.quality_metrics.*
uc_metadata_assistant.cache.*
```

Tables:
- `metadata_results` - Generated descriptions
- `generation_audit` - Audit trail
- `completeness_snapshots` - Quality metrics history
- `pii_column_assessments` - PII analysis results
- `workspace_settings` - App configuration

## Technology Stack

- **Backend:** Flask, Python 3.9+
- **Frontend:** React, Vite, TailwindCSS, Recharts, Framer Motion
- **Database:** Unity Catalog (Delta Lake)
- **AI/ML:** Databricks Foundation Model Serving
- **Infrastructure:** Databricks Apps Platform

## API Endpoints

### Metadata
- `GET /api/catalogs` - List available catalogs
- `GET /api/fast-counts/<catalog>` - Metadata counts
- `GET /api/missing-metadata/<catalog>/<type>` - Missing metadata
- `GET /api/coverage/<catalog>` - Coverage trends

### Generation
- `POST /api/enhanced/run` - Start generation
- `GET /api/enhanced/status/<run_id>` - Generation status
- `POST /api/enhanced/cancel/<run_id>` - Cancel generation
- `GET /api/enhanced/results` - Retrieve results

### Metadata Copy & Sync
- `GET /api/metadata/source-objects` - Fetch objects with descriptions
- `POST /api/metadata/smart-match` - Generate smart matches
- `POST /api/metadata/copy-bulk` - Apply bulk metadata copy
- `POST /api/metadata/export` - Export metadata to CSV
- `POST /api/metadata/import` - Parse and validate CSV import
- `POST /api/metadata/import-apply` - Apply imported metadata

### Quality
- `GET /api/quality-metrics/<catalog>` - Quality dashboard data
- `GET /api/metadata-history/<catalog>` - Audit history

### Settings
- `GET/POST /api/settings/models` - Model configuration
- `GET/POST /api/settings/sensitive-data` - PII configuration
- `GET/POST /api/settings/tags` - Tag policy
- `GET/POST /api/settings/sampling` - Sampling configuration
- `GET/POST /api/settings/prompts` - Prompt customization
- `GET/POST /api/settings/generation-mode` - Generation engine configuration
- `POST /api/dbxmetagen/validate` - Validate dbxmetagen setup

### Unity Catalog
- `POST /api/submit-metadata` - Commit to Unity Catalog
- `GET /api/governed-tags` - Fetch governed tags

## Performance

Generation time varies by:
- LLM model (Gemma fastest, Claude slowest)
- Batch size (configurable 1-50 objects)
- PII detection settings (pattern-only vs. LLM-enhanced)
- Endpoint availability

## Advanced Workflows

### Metadata Copy & Sync

The Copy Metadata tab enables seamless metadata promotion and cross-workspace synchronization:

**Smart Matching:**
- Automatically detects bronze→silver→gold patterns
- Fuzzy name matching for schema/table/column renaming
- Confidence scoring (exact: 100%, pattern: 90%, fuzzy: 70-90%)
- Manual mapping for edge cases

**Use Cases:**
1. **Pipeline Promotion:** Copy descriptions from `sales.bronze` to `sales.silver` as tables are cleansed
2. **Cross-Workspace Sync:** Export from Dev workspace, import to Test/Prod
3. **Bulk Updates:** Copy descriptions from well-documented tables to similar tables

**CSV Format for Descriptions:**
```csv
full_name,object_type,description
sales.bronze,schema,"Bronze layer for raw ingestion"
sales.bronze.orders,table,"Raw orders from Salesforce"
sales.bronze.orders.order_id,column,"Unique order identifier"
```

**CSV Format with Tags:**
```csv
full_name,object_type,description,tag_key,tag_value
sales.bronze,schema,"Bronze layer for raw ingestion",Environment,Production
sales.bronze,schema,"Bronze layer for raw ingestion",Owner,DataEng
sales.bronze.orders,table,"Raw orders from Salesforce",Domain,Sales
sales.bronze.orders.customer_id,column,"Customer identifier",PII,customer_id
```

**Multiple Tags for Single Object:**
- Each tag requires a separate row with the same `full_name`
- Or use comma-separated values: `"PII,Sensitive"`
- Tags are validated against governed tags if enabled

### dbxmetagen Integration

UC Metadata Assistant can operate as a UI/orchestration layer for the [dbxmetagen industry solution](https://github.com/technical-solutions-lakehouse/dbxmetagen):

**When to Use dbxmetagen:**
- Require Presidio-based PII detection (spaCy + custom patterns)
- Need medical information classification (HIPAA compliance)
- Prefer Excel/TSV review workflows
- Want MLflow-based prompt management
- Have existing dbxmetagen deployments

**Setup:**
1. Deploy dbxmetagen notebooks to your workspace
2. Configure variables.yml in DBFS
3. Go to Settings → Generation Mode
4. Select "dbxmetagen" and configure:
   - Notebook path: `/Workspace/Shared/dbxmetagen/notebooks/metadata_generator`
   - Cluster ID: Your cluster ID
   - Variables path: `/dbfs/dbxmetagen/variables.yml`
5. Click "Test Connection" to validate
6. Save and generate metadata normally

**Unified Experience:**
- Same UI for both engines
- Results imported into UC Metadata Assistant tables
- Review & Commit workflow works identically
- Quality metrics track both sources

**Engine Comparison:**

| Feature | UC Assistant | dbxmetagen |
|---------|-------------|------------|
| **Speed** | Fast (real-time) | Moderate (batch job) |
| **PII Detection** | Pattern + LLM | Presidio (spaCy) |
| **Medical Info** | General PII | HIPAA-focused |
| **Progress Tracking** | Real-time UI | Job logs |
| **Customization** | Prompt config UI | variables.yml |
| **Output** | Direct to UC | Excel/TSV → UC |
| **Best For** | Interactive use | Compliance-heavy |

## Troubleshooting

### Setup Issues

#### "App failed to start" or blank screen
**Cause**: Missing or incorrect warehouse configuration in `app.yaml`

**Solution**:
1. Verify `DATABRICKS_HTTP_PATH` and `DATABRICKS_SERVER_HOSTNAME` are correct
2. Ensure SQL warehouse is running
3. Check app logs in Databricks Apps UI

#### Catalog Creation Fails
**Cause**: "Metastore storage root URL does not exist" or catalog `uc_metadata_assistant` doesn't exist

**Solution**:
1. **Recommended**: Pre-create the catalog manually (Step 2 from setup):
   ```sql
   CREATE CATALOG IF NOT EXISTS uc_metadata_assistant;
   ```
2. Grant necessary permissions:
   ```sql
   GRANT ALL PRIVILEGES ON CATALOG uc_metadata_assistant TO `<app-service-principal>`;
   ```
3. Restart or redeploy the app

**Alternative** (if you want auto-creation):
- Grant `CREATE CATALOG ON METASTORE` to the service principal
- This is not required if you follow Step 2 of the setup guide

#### Cannot Find Service Principal
**Cause**: App not yet deployed or name unclear

**Solution**:
1. Go to **Apps** → **Your App** → **Configuration** tab
2. Look for "Service Principal" section
3. Name format: `app-<app-name>-<random-id>`
4. Or use CLI: `databricks apps get <app-name>`

### Permission Errors

#### "Permission denied: USE CATALOG"
**Cause**: App service principal lacks catalog access

**Solution**:
```sql
GRANT USE CATALOG ON CATALOG <catalog-name> TO `<app-service-principal>`;
GRANT USE SCHEMA ON SCHEMA <catalog-name>.* TO `<app-service-principal>`;
GRANT SELECT ON TABLE <catalog-name>.*.* TO `<app-service-principal>`;
```

#### "Cannot write COMMENT" or "MODIFY required"
**Cause**: Missing MODIFY permission for description generation

**Solution**:
```sql
GRANT MODIFY ON CATALOG <catalog-name> TO `<app-service-principal>`;
GRANT MODIFY ON SCHEMA <catalog-name>.* TO `<app-service-principal>`;
GRANT MODIFY ON TABLE <catalog-name>.*.* TO `<app-service-principal>`;
```

#### "Cannot apply tag" or "APPLY TAG required"
**Cause**: Missing tag permissions for PII detection or CSV import

**Solution**:
```sql
GRANT APPLY TAG ON CATALOG <catalog-name> TO `<app-service-principal>`;
GRANT APPLY TAG ON SCHEMA <catalog-name>.* TO `<app-service-principal>`;
GRANT APPLY TAG ON TABLE <catalog-name>.*.* TO `<app-service-principal>`;
```

#### "Governed tag validation failed"
**Cause**: Cannot read tag definitions from `system.information_schema`

**Solution**:
```sql
GRANT USE CATALOG ON CATALOG system TO `<app-service-principal>`;
GRANT SELECT ON SCHEMA system.information_schema TO `<app-service-principal>`;
```

#### "SQL warehouse access denied"
**Cause**: Missing warehouse permissions

**Solution**:
```sql
GRANT USE ON SQL WAREHOUSE <warehouse-id> TO `<app-service-principal>`;
```

#### "Model endpoint unavailable"
**Cause**: Missing serving endpoint permissions or endpoint is disabled

**Solution**:
1. Check endpoint status in **Serving** UI
2. Grant access:
   ```sql
   GRANT EXECUTE ON ANY FILE TO `<app-service-principal>`;
   ```
3. Or per-model: `GRANT EXECUTE ON MODEL <endpoint> TO service-principal`

### Runtime Issues

#### Slow Performance
**Causes**: Warehouse size, batch size, or network latency

**Solutions**:
- Use Pro or Serverless SQL warehouse
- Reduce batch size in generation config
- Enable sampling (Settings → Sampling) to limit row reads
- Check warehouse scaling settings

#### No Catalogs Visible
**Causes**: Missing `USE CATALOG` permission or warehouse connectivity

**Solutions**:
1. Test warehouse connectivity: `SELECT current_catalog()`
2. Grant `USE CATALOG` on all target catalogs
3. Refresh the app page

#### PII Detection Not Working
**Causes**: Disabled in settings, missing patterns, or tag conflicts

**Solutions**:
1. Settings → Sensitive Data → Verify PII detection is enabled
2. Check detection mode (pattern-only vs. LLM-enhanced)
3. Settings → Tags → Verify tag mappings are configured
4. Review app logs for governed tag conflicts

#### Metadata Generation Fails
**Causes**: Model endpoint issues, permission errors, or prompt problems

**Solutions**:
1. Check model endpoint status (Settings → Models)
2. Verify `MODIFY` permissions on target catalog
3. Review app logs for specific error messages
4. Try a different model (some endpoints may be rate-limited)

#### Tags Failed to Import
**Causes**: Governed tag conflicts or permission errors

**Solutions**:
1. Use "Validate Tags" button after CSV import
2. Check if tag values match governed tag allowed values
3. Verify `APPLY TAG` permissions
4. Review error message in import result banner

### Data Issues

#### Sample Data Not Appearing
**Cause**: Sampling disabled or table has no data

**Solution**:
1. Settings → Sampling → Enable data sampling
2. Set sample row count (recommended: 5-10 rows)
3. Verify table has data: `SELECT * FROM table LIMIT 5`

#### Incorrect Coverage Metrics
**Cause**: Cached data or incomplete metadata scan

**Solution**:
1. Navigate to Overview tab
2. Click "Refresh Data" to rescan catalog
3. Wait for completeness snapshot to update

#### CSV Import Shows "0 valid"
**Causes**: Incorrect CSV format or column names

**Solution**:
1. Verify CSV has required columns: `full_name`, `object_type`, `description`
2. Use correct full names: `catalog.schema.table.column`
3. Valid object types: `schema`, `table`, `column`
4. Download example CSV from import UI

### App Logs

To view detailed error messages:
1. Go to **Apps** → **Your App**
2. Click **Logs** tab
3. Filter by "ERROR" or "WARNING"
4. Look for timestamps matching your operation

Common log messages:
- `✅` = Success
- `❌` = Error
- `⚠️` = Warning
- `🔍` = Discovery/validation
- `📝` = Metadata operation

## Development

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABRICKS_HOST=<workspace-url>
export DATABRICKS_TOKEN=<token>
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>

# Run locally
python app_react.py
```

### Frontend Development

```bash
cd client
npm install
npm run dev  # Development server
npm run build  # Production build
```

## Security

- All data processing happens within Databricks
- No external API calls or data exports
- Service principal authentication with OAuth2
- Unity Catalog permission validation
- Encrypted communication via HTTPS/TLS
- Complete audit trails for compliance

## License

MIT License - see LICENSE file for details.

## Support

- GitHub Issues: Bug reports and feature requests
- Databricks Community: General questions
- Enterprise Support: Priority support for production deployments
