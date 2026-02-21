-- Add authentication fields to clients table
ALTER TABLE clients 
ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255),
ADD COLUMN IF NOT EXISTS current_session_token VARCHAR(512),
ADD COLUMN IF NOT EXISTS session_created_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS session_expires_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS login_count INTEGER DEFAULT 0;

-- Create index for faster session lookups
CREATE INDEX IF NOT EXISTS idx_clients_session_token ON clients(current_session_token);

-- Verify columns
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'clients' 
  AND column_name IN ('password_hash', 'current_session_token', 'session_expires_at');