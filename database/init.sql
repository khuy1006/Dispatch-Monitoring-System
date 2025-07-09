-- Initialize database for production PostgreSQL setup
-- This file is executed when the PostgreSQL container starts

-- Create monitoring database if it doesn't exist
-- CREATE DATABASE monitoring;

-- Connect to monitoring database
\c monitoring;

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_id VARCHAR(255),
    original_class VARCHAR(100),
    correct_class VARCHAR(100),
    confidence REAL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    user_feedback TEXT,
    processed BOOLEAN DEFAULT FALSE
);

-- Create analytics table
CREATE TABLE IF NOT EXISTS analytics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(100),
    metric_value REAL,
    additional_data JSONB
);

-- Create detection_logs table for tracking all detections
CREATE TABLE IF NOT EXISTS detection_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_id VARCHAR(255),
    class_name VARCHAR(100),
    confidence REAL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    track_id INTEGER,
    classification_result VARCHAR(100),
    classification_confidence REAL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback(processed);
CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp);
CREATE INDEX IF NOT EXISTS idx_analytics_metric ON analytics(metric_name);
CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_detection_logs_class ON detection_logs(class_name);

-- Create user for the application
CREATE USER monitoring_user WITH PASSWORD 'monitoring_pass';
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO monitoring_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO monitoring_user;
