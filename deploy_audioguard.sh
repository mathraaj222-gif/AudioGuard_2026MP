#!/bin/bash

# --- CONFIG ---
PROJECT_ID=$(gcloud config get-value project)
REGION="asia-southeast1"
REPO="audioguard-repo"
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

echo "🚀 Starting AudioGuard 2026 Cloud Deployment..."
echo "Project: ${PROJECT_ID} | Region: ${REGION}"

# 1. Enable APIs
echo "📡 Enabling Google Cloud APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 2. Create Registry
echo "📦 Creating Artifact Registry..."
gcloud artifacts repositories create ${REPO} --repository-format=docker --location=${REGION} --description="AudioGuard Microservices" || true

# 3. Clone Latest Code
echo "📂 Cloning Repository..."
rm -rf AudioGuard_2026MP
git clone https://github.com/mathraaj222-gif/AudioGuard_2026MP.git
cd AudioGuard_2026MP

# 4. Build Images (Parallel via Cloud Build)
echo "🏗️ Building Microservice Containers in the Cloud..."
gcloud builds submit --tag ${IMAGE_BASE}/whisper-svc ./ml_services/whisper-svc &
gcloud builds submit --tag ${IMAGE_BASE}/ser-svc ./ml_services/ser-svc &
gcloud builds submit --tag ${IMAGE_BASE}/tca-svc ./ml_services/tca-svc &
gcloud builds submit --tag ${IMAGE_BASE}/backend ./backend &
wait

# 5. Deploy AI Services (To get URLs)
echo "🚀 Deploying AI Microservices..."
gcloud run deploy whisper-svc --image ${IMAGE_BASE}/whisper-svc --region ${REGION} --platform managed --memory 4Gi --cpu 2 --allow-unauthenticated --no-cpu-throttling --quiet
WHISPER_URL=$(gcloud run services describe whisper-svc --region ${REGION} --format 'value(status.url)')

gcloud run deploy ser-svc --image ${IMAGE_BASE}/ser-svc --region ${REGION} --platform managed --memory 2Gi --cpu 1 --allow-unauthenticated --quiet
SER_URL=$(gcloud run services describe ser-svc --region ${REGION} --format 'value(status.url)')

gcloud run deploy tca-svc --image ${IMAGE_BASE}/tca-svc --region ${REGION} --platform managed --memory 1Gi --cpu 1 --allow-unauthenticated --quiet
TCA_URL=$(gcloud run services describe tca-svc --region ${REGION} --format 'value(status.url)')

# 6. Deploy Backend Orchestrator (With Linked URLs)
echo "🔗 Linking Services & Deploying Orchestrator..."
gcloud run deploy audioguard-backend \
  --image ${IMAGE_BASE}/backend \
  --region ${REGION} \
  --platform managed \
  --memory 1Gi \
  --allow-unauthenticated \
  --set-env-vars "WHISPER_URL=${WHISPER_URL},SER_URL=${SER_URL},TCA_URL=${TCA_URL}" \
  --quiet

FINAL_URL=$(gcloud run services describe audioguard-backend --region ${REGION} --format 'value(status.url)')

echo "----------------------------------------------------"
echo "✅ DEPLOYMENT COMPLETE!"
echo "Backend URL: ${FINAL_URL}"
echo "----------------------------------------------------"
