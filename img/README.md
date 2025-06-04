# Deployment Architecture Diagrams

This directory should contain the following diagram files:

- `deployment-architecture.png`: Main deployment architecture diagram showing the different deployment modes and platform combinations.

The diagrams can be created using tools like:
- [draw.io](https://draw.io) (recommended)
- [mermaid.js](https://mermaid.js.org/)
- [PlantUML](https://plantuml.com/)

## Diagram Elements to Include

The deployment architecture diagram should show:

1. The three deployment modes:
   - Full mode (frontend and backend together)
   - API-only mode (backend only)
   - UI-only mode (frontend only)

2. The different platform combinations:
   - NVIDIA LaunchPad backend + Vercel frontend
   - Together.ai backend + Vercel frontend
   - SAP BTP backend + SAP BTP frontend
   - SAP BTP backend + Vercel frontend
   - Together.ai backend + SAP BTP frontend

3. The communication flows between components, including:
   - API requests
   - CORS configuration
   - Failover between backends

4. Key components:
   - Frontend UI
   - Backend API
   - HANA database connections
   - NVIDIA GPU optimization
   - Together.ai API integration
   - Deployment configuration generator

## Color Scheme

Use a consistent color scheme:
- Blue for SAP BTP components
- Green for NVIDIA components
- Purple for Together.ai components
- Orange for Vercel components
- Gray for connections and communication flows