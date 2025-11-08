from flask import Blueprint, request, jsonify, send_file
import logging
from werkzeug.utils import secure_filename

from services.image_service import image_service

images_bp = Blueprint('images', __name__, url_prefix='/api/images')

@images_bp.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image file"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read file data
        file_data = file.read()
        filename = secure_filename(file.filename)
        
        success, file_path, error = image_service.save_uploaded_file(file_data, filename)
        
        if success:
            # Get image info
            image_info = image_service.get_image_info(file_path)
            
            return jsonify({
                'success': True,
                'file_path': file_path,
                'filename': filename,
                'image_info': image_info,
                'message': 'Image uploaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to upload image'
            }), 400
            
    except Exception as e:
        logging.error(f"Error in upload_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@images_bp.route('/upload-base64', methods=['POST'])
def upload_base64_image():
    """Upload an image from base64 data"""
    try:
        data = request.get_json()
        
        if not data or 'base64Data' not in data or 'filename' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing base64Data or filename'
            }), 400
        
        base64_data = data['base64Data']
        filename = secure_filename(data['filename'])
        
        success, file_path, error = image_service.save_base64_image(base64_data, filename)
        
        if success:
            # Get image info
            image_info = image_service.get_image_info(file_path)
            
            return jsonify({
                'success': True,
                'file_path': file_path,
                'filename': filename,
                'image_info': image_info,
                'message': 'Image uploaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': error or 'Failed to upload image'
            }), 400
            
    except Exception as e:
        logging.error(f"Error in upload_base64_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@images_bp.route('/<path:image_path>', methods=['GET'])
def serve_image(image_path: str):
    """Serve a stored image file"""
    try:
        # Ensure the path is safe and within the images directory
        if not image_path.startswith('images/'):
            image_path = f'images/{image_path}'
        
        file_path = image_service.get_image_path(image_path)
        
        if file_path and file_path.exists():
            return send_file(
                file_path,
                as_attachment=False,
                download_name=file_path.name
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
            
    except Exception as e:
        logging.error(f"Error serving image {image_path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@images_bp.route('/<path:image_path>/base64', methods=['GET'])
def get_image_base64(image_path: str):
    """Get image as base64 data"""
    try:
        # Ensure the path is safe and within the images directory
        if not image_path.startswith('images/'):
            image_path = f'images/{image_path}'
        
        base64_data = image_service.get_image_base64(image_path)
        
        if base64_data:
            return jsonify({
                'success': True,
                'base64Data': base64_data,
                'image_path': image_path
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
            
    except Exception as e:
        logging.error(f"Error getting base64 for image {image_path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@images_bp.route('/<path:image_path>/info', methods=['GET'])
def get_image_info(image_path: str):
    """Get information about a stored image"""
    try:
        # Ensure the path is safe and within the images directory
        if not image_path.startswith('images/'):
            image_path = f'images/{image_path}'
        
        image_info = image_service.get_image_info(image_path)
        
        if image_info:
            return jsonify({
                'success': True,
                'image_info': image_info,
                'image_path': image_path
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
            
    except Exception as e:
        logging.error(f"Error getting info for image {image_path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@images_bp.route('/<path:image_path>', methods=['DELETE'])
def delete_image(image_path: str):
    """Delete a stored image"""
    try:
        # Ensure the path is safe and within the images directory
        if not image_path.startswith('images/'):
            image_path = f'images/{image_path}'
        
        success = image_service.delete_image(image_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Image deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete image or image not found'
            }), 404
            
    except Exception as e:
        logging.error(f"Error deleting image {image_path}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@images_bp.route('/cleanup', methods=['POST'])
def cleanup_orphaned_images():
    """Remove images that are no longer referenced"""
    try:
        # This endpoint would typically be called by admin
        # For now, we'll just return a placeholder response
        # In a real implementation, you'd get all image paths from ChromaDB
        # and pass them to image_service.cleanup_orphaned_images()
        
        return jsonify({
            'success': True,
            'message': 'Cleanup functionality available but requires admin authentication',
            'deleted_count': 0
        })
        
    except Exception as e:
        logging.error(f"Error in cleanup_orphaned_images: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500