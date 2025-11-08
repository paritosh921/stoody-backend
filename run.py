#!/usr/bin/env python3
"""
SkillBot Backend Server Runner

This script starts the Flask backend server for the SkillBot application.
It handles ChromaDB storage for questions and file storage for images.

Usage:
    python run.py              # Start with default settings
    python run.py --debug      # Start with debug mode
    python run.py --port 5001  # Start on custom port
"""

import sys
import argparse
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

from app import create_app
from config import FLASK_PORT, FLASK_DEBUG

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SkillBot Backend Server')
    parser.add_argument(
        '--port', 
        type=int, 
        default=FLASK_PORT,
        help=f'Port to run the server on (default: {FLASK_PORT})'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        default=FLASK_DEBUG,
        help='Run in debug mode'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create Flask app
    app = create_app()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    SkillBot Backend API                      ║
║                                                              ║
║  🚀 Starting Flask server...                                ║
║  📍 URL: http://{args.host}:{args.port}                            
║  🔧 Debug mode: {'ON' if args.debug else 'OFF'}                                     ║
║  📚 API Documentation: http://{args.host}:{args.port}/                    
║                                                              ║
║  📁 Data Storage:                                            ║
║     • ChromaDB: {backend_dir}/chromadb_data                    
║     • Images: {backend_dir}/images                             
║                                                              ║
║  🔌 CORS enabled for:                                        ║
║     • http://localhost:8080 (Frontend)                      ║
║     • http://127.0.0.1:8080                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down SkillBot Backend API...")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()