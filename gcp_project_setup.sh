#!/bin/bash

# Exit on any error
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_message() {
    echo -e "${GREEN}[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%dT%H:%M:%S%z')] WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date +'%Y-%m-%dT%H:%M:%S%z')] ERROR: $1${NC}"
}

# Install Terraform if not installed
install_terraform() {
    if ! command -v terraform &> /dev/null; then
        print_message "Installing Terraform..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            print_message "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install Terraform using Homebrew
        brew tap hashicorp/tap
        brew install hashicorp/tap/terraform
        
        # Verify installation
        terraform --version
        
        print_message "Terraform installed successfully!"
    else
        print_message "Terraform is already installed."
    fi
}

# Check GCP authentication
check_gcp_auth() {
    print_message "Checking GCP authentication..."
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q "@"; then
        print_message "Please authenticate with Google Cloud..."
        gcloud auth login
        
        # Set up application default credentials
        print_message "Setting up application default credentials..."
        gcloud auth application-default login
    else
        print_message "Already authenticated with GCP."
    fi
    
    # Verify project ID is set
    if [ -z "$PROJECT_ID" ]; then
        print_error "PROJECT_ID environment variable is not set"
        exit 1
    fi
    
    # Set the project
    print_message "Setting GCP project to: $PROJECT_ID"
    gcloud config set project $PROJECT_ID
    
    # Set application default credentials quota project
    print_message "Setting quota project..."
    gcloud auth application-default set-quota-project $PROJECT_ID
}

# Check if required commands are installed
check_requirements() {
    print_message "Checking requirements..."
    
    commands=("gcloud" "python3" "pip3" "git")
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is required but not installed."
            exit 1
        fi
    done
    
    # Install Terraform separately
    install_terraform
    
    print_message "All requirements satisfied."
}

# Setup Python environment
setup_python_env() {
    print_message "Setting up Python environment..."
    
    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        print_message "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create virtual environment with Python 3.9
    print_message "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    print_message "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_message "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_message "Installing requirements..."
    pip install -r requirements.txt
}

# Initialize GCP project
init_gcp_project() {
    print_message "Initializing GCP project..."
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        print_error "PROJECT_ID environment variable is not set"
        exit 1
    fi
    
    # Configure gcloud
    print_message "Configuring gcloud..."
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    print_message "Enabling required APIs..."
    apis=(
        "cloudfunctions.googleapis.com"
        "cloudbuild.googleapis.com"
        "bigquery.googleapis.com"
        "aiplatform.googleapis.com"
        "storage.googleapis.com"
        "pubsub.googleapis.com"
        "cloudscheduler.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        print_message "Enabling $api..."
        gcloud services enable $api
    done
}

# Setup Terraform
setup_terraform() {
    print_message "Setting up Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    print_message "Initializing Terraform..."
    terraform init
    
    # Apply Terraform configuration
    print_message "Applying Terraform configuration..."
    terraform apply -var="project_id=$PROJECT_ID" -auto-approve
    
    cd ..
}

# Deploy Cloud Functions
deploy_functions() {
    print_message "Deploying Cloud Functions..."
    
    # Submit Cloud Build job
    print_message "Submitting Cloud Build job..."
    gcloud builds submit --config cloudbuild.yaml
}

# Create initial BigQuery tables
setup_bigquery() {
    print_message "Setting up BigQuery tables..."
    
    # Create dataset if it doesn't exist
    bq mk --dataset \
        --description "Airbnb Analytics Dataset" \
        --location $REGION \
        $PROJECT_ID:airbnb_analytics
}

# Setup monitoring
setup_monitoring() {
    print_message "Setting up Cloud Monitoring..."
    
    # Create monitoring workspace
    gcloud monitoring workspaces create \
        --display-name="Airbnb Analytics Monitoring" \
        --location=$REGION
}

# Main setup process
main() {
    print_message "Starting GCP project setup..."
    
    # Check requirements
    check_requirements
    
    # Check GCP authentication
    check_gcp_auth
    
    # Initialize GCP project
    init_gcp_project
    
    # Setup Python environment
    setup_python_env
    
    # Setup Terraform infrastructure
    setup_terraform
    
    # Deploy Cloud Functions
    deploy_functions
    
    # Setup BigQuery
    setup_bigquery
    
    # Setup monitoring
    setup_monitoring
    
    print_message "Setup completed successfully!"
    print_message "Next steps:"
    print_message "1. Upload your raw data to gs://$PROJECT_ID-raw-data/"
    print_message "2. Monitor the pipeline in Cloud Console"
    print_message "3. Check BigQuery for processed data"
    print_message "4. View models in Vertex AI"
}

# Set your GCP project
export PROJECT_ID='lateral-vision-438701-u5'

# Execute main function
main