/// Upload screen — trigger upload, select path, monitor progress, camera.
///
/// Allows the invigilator to:
///   1. Choose upload path (wifi / mobile / usb).
///   2. Trigger the hub's upload process (cmd 0x05).
///   3. Monitor upload progress via the status feed.
///   4. Capture a photo of the answer sheet as a fallback (camera).
library;

import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../ble_service.dart';
import '../models/hub_status.dart';
import '../widgets/sync_progress_bar.dart';

/// Upload path options — must match hub's accepted values.
enum UploadPath {
  wifi('wifi'),
  mobile('mobile'),
  usb('usb');

  const UploadPath(this.wireValue);
  final String wireValue;
}

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final BleService _ble = BleService();

  HubStatus? _status;
  StreamSubscription<HubStatus>? _statusSub;

  UploadPath _selectedPath = UploadPath.wifi;
  bool _uploading = false;
  String? _error;

  // Camera
  CameraController? _cameraController;
  bool _cameraReady = false;
  bool _capturing = false;
  String? _lastPhotoPath;

  @override
  void initState() {
    super.initState();
    _statusSub = _ble.statusStream.listen((s) {
      if (mounted) setState(() => _status = s);
    });
  }

  @override
  void dispose() {
    _statusSub?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Upload trigger
  // ---------------------------------------------------------------------------

  Future<void> _triggerUpload() async {
    setState(() {
      _uploading = true;
      _error = null;
    });

    try {
      final requestId = DateTime.now().millisecondsSinceEpoch
          .toRadixString(36)
          .padLeft(16, '0');
      await _ble.startUpload(
        examId: _status?.examId ?? '',
        path: _selectedPath.wireValue,
        requestId: requestId,
      );
    } catch (e) {
      if (mounted) setState(() => _error = 'Upload failed: $e');
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  // ---------------------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------------------

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      if (mounted) setState(() => _error = 'No camera available');
      return;
    }

    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      if (mounted) setState(() => _cameraReady = true);
    } catch (e) {
      if (mounted) setState(() => _error = 'Camera init failed: $e');
    }
  }

  Future<void> _capturePhoto() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    setState(() => _capturing = true);

    try {
      final file = await _cameraController!.takePicture();
      if (mounted) {
        setState(() {
          _lastPhotoPath = file.path;
          _capturing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Capture failed: $e';
          _capturing = false;
        });
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final syncSummary = _status?.sync;

    return Scaffold(
      appBar: AppBar(title: const Text('Upload')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Upload progress
            if (syncSummary != null) ...[
              Text(
                'Sync: ${syncSummary.complete}/${syncSummary.total} pens',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              SyncProgressBar(fraction: syncSummary.progressFraction),
              const SizedBox(height: 24),
            ],

            // Path selector
            Text(
              'Upload Path',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            SegmentedButton<UploadPath>(
              segments: const [
                ButtonSegment(
                  value: UploadPath.wifi,
                  label: Text('WiFi'),
                  icon: Icon(Icons.wifi),
                ),
                ButtonSegment(
                  value: UploadPath.mobile,
                  label: Text('Mobile'),
                  icon: Icon(Icons.phone_android),
                ),
                ButtonSegment(
                  value: UploadPath.usb,
                  label: Text('USB'),
                  icon: Icon(Icons.usb),
                ),
              ],
              selected: {_selectedPath},
              onSelectionChanged: (s) =>
                  setState(() => _selectedPath = s.first),
            ),
            const SizedBox(height: 24),

            // Upload button
            ElevatedButton.icon(
              onPressed: _uploading ? null : _triggerUpload,
              icon: _uploading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.cloud_upload),
              label: const Text('Start Upload'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),

            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(
                _error!,
                style: const TextStyle(color: Colors.red),
                textAlign: TextAlign.center,
              ),
            ],

            const SizedBox(height: 32),
            const Divider(),
            const SizedBox(height: 16),

            // Camera fallback
            Text(
              'Fallback: Photo Capture',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 12),

            if (!_cameraReady) ...[
              ElevatedButton.icon(
                onPressed: _initCamera,
                icon: const Icon(Icons.camera_alt),
                label: const Text('Open Camera'),
              ),
            ] else ...[
              AspectRatio(
                aspectRatio: 3 / 4,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: CameraPreview(_cameraController!),
                ),
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                onPressed: _capturing ? null : _capturePhoto,
                icon: _capturing
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.camera),
                label: const Text('Capture'),
              ),
            ],

            if (_lastPhotoPath != null) ...[
              const SizedBox(height: 12),
              Text(
                'Saved: $_lastPhotoPath',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey.shade600,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
