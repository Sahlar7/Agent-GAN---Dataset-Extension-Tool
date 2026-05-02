# Datagent - AI-Powered Dataset Extension Tool

An intelligent tool for extending image datasets using LLM-built Generative Adversarial Networks (GANs). Datagent automates the process of generating synthetic data to augment existing datasets, with integration for remote HPC training and MCP-based agent orchestration.

Based on a natural langauge prompt, and dataset information either provided by the user or inferred via MCP dataset exploration tools, an agent selects architectural and training parameters (dimensions, activation, loss, decay, learning rate, betas, epochs, batch size, + more) to train a GAN to extend the provided dataset. The Universal GAN classd adapts the selected parameters to create a PyTorch GAN architecture, which is then trained using a generated train script based on a provided templated. The train script is submitted with a slurm script to the University of Virginia's (UVA) HPC system, Rivanna (credentials required).

## 🎯 Project Overview

Datagent combines:
- **GAN-based Image Generation**: Generate synthetic images to extend datasets
- **AI Agent Orchestration**: Use LangChain agent with an MCP server for dataset analysis. The agent selects GAN architectural parameters, writes a Pytorch training script, and submits a slurm job to UVA Rivanna HPC
- **REST API Backend**: FastAPI-based service for job management and data processing
- **Web UI**: React + Vite frontend for uploading dataset/natural language to prompt agent
- **Remote Training**: Submit jobs to HPC systems (UVA Rivanna HPC)

## 🏗️ Architecture

```
Backend (Python/FastAPI)
├── API Routes - Job management and endpoints
├── Agent Service - AI agent for workflow orchestration
├── Job Service - Job queue and status tracking
├── MCP Server - Model Context Protocol server dataset analysis tool and job submission
└── GAN Utilities - Training scripts and univeral GAN model architecture (adaptive to agent selected parameters)

Frontend (React/Vite)
├── Job Upload Screen - Submit datasets for processing
├── Status Monitoring - Track job progress
└── Result Management - View and download generated data
```

## 📋 Prerequisites

- **Python 3.9+** (for backend)
- **Node.js 18+** (for frontend)
- **pip** (Python package manager)
- **npm** or **yarn** (Node package manager)
- **.env file** with required API keys (see Configuration section)

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv ../venv

# Activate virtual environment
# On Windows:
../venv\Scripts\activate
# On macOS/Linux:
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file in backend directory
# Add required configuration (see Configuration section)

# Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8100 --reload
```

The backend API will be available at `http://localhost:8100`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd client/react-client

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 3. MCP Server Setup

```bash
# Navigate to backend directory (if not already there)
cd backend

# Ensure virtual environment is activated
# On Windows:
../venv\Scripts\activate
# On macOS/Linux:
source ../venv/bin/activate

# Start the MCP server (in a separate terminal)
python -m mcp_server.run_mcp_server
```

The MCP server will be available on the port specified in your `.env` file (default: 8101)

**Note**: The MCP server should run in a separate terminal from the API server. This server handles:
- Dataset analysis tools
- GAN parameter selection
- Job submission to HPC systems
- Integration with the AI agent

## ⚙️ Configuration

### Backend Configuration

Create a `.env` file in the `backend` directory:

```env
# Google Generative AI
GOOGLE_API_KEY=your_google_api_key_here

# Optional: HPC/Rivanna Configuration
HPC_HOST=rivanna.hpc.university.edu
HPC_USER=your_username
HPC_PASSWORD=your_password

# Optional: MCP Server Configuration
MCP_PORT=8101
```

### Frontend Configuration

The frontend is configured to connect to the backend API on:
- Development: `http://localhost:8100`
- Production: Update in `vite.config.ts` or environment variables

## 📁 Project Structure

```
datagentic-GAN/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Configuration and constants
│   │   └── routes/
│   │       └── jobs.py          # Job management endpoints
│   ├── services/
│   │   ├── agent_service.py     # AI agent orchestration
│   │   └── job_service.py       # Job queue management
│   ├── agent/
│   │   ├── agent.py             # LangChain agent definition
│   ├── mcp_server/
│   │   ├── run_mcp_server.py    # MCP server and tools
│   │   └── utils/
│   │       ├── train_gan.py     # GAN training logic
│   │       ├── universal_gan.py # Universal GAN model
│   │       └── rivanna.py       # HPC integration
│   ├── uploads/                 # User uploaded datasets
│   ├── requirements.txt         # Python dependencies
│   └── __init__.py
├── client/react-client/
│   ├── src/
│   │   ├── App.tsx              # Main React component
│   │   ├── components/          # React components
│   │   ├── main.tsx             # React entry point
│   │   └── index.css            # Global styles
│   ├── package.json             # Node dependencies
│   ├── vite.config.ts           # Vite configuration
│   ├── tsconfig.json            # TypeScript configuration
│   └── index.html
└── README.md                    # This file
```

## 🔧 Key Features

### Job Management
- Upload image datasets for processing
- Track job status and progress
- Download generated synthetic data
- Job history and logs

### AI Agent
- LangChain-based intelligent agent
- Google Generative AI integration
- MCP tools for dataset analysis so agent can infer better
GAN architecture
- Chainable operations for complex data pipelines

### GAN Training
- Support for multiple GAN architectures
- Local and remote training capabilities
- Model checkpoint management
- Configurable training parameters

### MCP Server
- Tool execution framework
- Remote job submission
- Model Context Protocol support


## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Generative AI API key | Yes |
| `HPC_HOST` | HPC server hostname | No |
| `HPC_USER` | HPC login username | No |
| `HPC_PASSWORD` | HPC login password | No |
| `MCP_PORT` | MCP server port | No (default: 8101) |

