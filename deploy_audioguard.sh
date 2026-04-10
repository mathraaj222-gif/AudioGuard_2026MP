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

# 3. Handle Code Sync
# If we are already inside the repo folder, don't clone again
if [[ "$PWD" == *"AudioGuard_2026MP"* ]] && [ -d ".git" ]; then
    echo "📂 Already in AudioGuard_2026MP repository. Skipping clone..."
else
    echo "📂 Cloning Repository fresh..."
    rm -rf AudioGuard_2026MP
    git clone https://github.com/mathraaj222-gif/AudioGuard_2026MP.git
    cd AudioGuard_2026MP
fi

# 4. Build Images (Parallel via Cloud Build)
echo "🏗️ Building Microservice Containers in the Cloud..."
gcloud builds submit --tag ${IMAGE_BASE}/whisper-svc ./ml_services/whisper-svc &
gcloud builds submit --tag ${IMAGE_BASE}/ser-svc ./ml_services/ser-svc &
gcloud builds submit --tag ${IMAGE_BASE}/tca-svc ./ml_services/tca-svc &
gcloud builds submit --tag ${IMAGE_BASE}/meta-svc ./ml_services/meta-svc &
gcloud builds submit --tag ${IMAGE_BASE}/backend ./backend &
wait

# Small delay to ensure Registry is synchronized before deployment starts
echo "⏳ Waiting for Registry synchronization..."
sleep 5

# 5. Deploy AI Services (To get URLs)
echo "🚀 Deploying AI Microservices..."
gcloud run deploy whisper-svc --image ${IMAGE_BASE}/whisper-svc --region ${REGION} --platform managed --memory 4Gi --cpu 2 --allow-unauthenticated --no-cpu-throttling
WHISPER_URL=$(gcloud run services describe whisper-svc --region ${REGION} --format 'value(status.url)')

gcloud run deploy ser-svc --image ${IMAGE_BASE}/ser-svc --region ${REGION} --platform managed --memory 2Gi --cpu 1 --allow-unauthenticated
SER_URL=$(gcloud run services describe ser-svc --region ${REGION} --format 'value(status.url)')

gcloud run deploy tca-svc --image ${IMAGE_BASE}/tca-svc --region ${REGION} --platform managed --memory 1Gi --cpu 1 --allow-unauthenticated
TCA_URL=$(gcloud run services describe tca-svc --region ${REGION} --format 'value(status.url)')

gcloud run deploy meta-svc --image ${IMAGE_BASE}/meta-svc --region ${REGION} --platform managed --memory 1Gi --cpu 1 --allow-unauthenticated
META_URL=$(gcloud run services describe meta-svc --region ${REGION} --format 'value(status.url)')

# 6. Deploy Backend Orchestrator (With Linked URLs)
echo "🔗 Linking Services & Deploying Orchestrator..."
gcloud run deploy audioguard-backend \
  --image ${IMAGE_BASE}/backend \
  --region ${REGION} \
  --platform managed \
  --memory 1Gi \
  --allow-unauthenticated \
  --set-env-vars "WHISPER_URL=${WHISPER_URL},SER_URL=${SER_URL},TCA_URL=${TCA_URL},META_URL=${META_URL},HATE_THRESHOLD=0.35"

FINAL_URL=$(gcloud run services describe audioguard-backend --region ${REGION} --format 'value(status.url)')

echo "----------------------------------------------------"
echo "✅ DEPLOYMENT COMPLETE!"
echo "Backend URL: ${FINAL_URL}"
echo "----------------------------------------------------"

echo "----------------------------------------------------"
echo "✅ DEPLOYMENT COMPLETE!"
echo "Backend URL: ${FINAL_URL}"
echo "----------------------------------------------------"
