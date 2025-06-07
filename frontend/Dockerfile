FROM node:16-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy source files
COPY . .

# Build the application
RUN npm run build

# Production stage with Nginx
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy Nginx configuration
COPY ./nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Set environment variables for the frontend
ENV API_URL=http://api:8000
ENV NAM_VISUALIZER_URL=http://nam-visualizer:8050
ENV NAM_DESIGN_THEME=light
ENV ENABLE_ANIMATIONS=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Expose port
EXPOSE 80
EXPOSE 443

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]