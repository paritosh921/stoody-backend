/// Hub BLE auth screen — scan for hubs, enter 12-char code, authenticate.
///
/// Flow:
///   1. Scan for nearby hubs advertising the invigilator service.
///   2. User taps a hub to connect.
///   3. Enter the 12-character invigilator code.
///   4. Write code to the Auth characteristic; display result.
///   5. On success, navigate to exam control.
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:go_router/go_router.dart';

import '../ble_service.dart';

class HubAuthScreen extends StatefulWidget {
  const HubAuthScreen({super.key});

  @override
  State<HubAuthScreen> createState() => _HubAuthScreenState();
}

class _HubAuthScreenState extends State<HubAuthScreen> {
  final BleService _ble = BleService();
  final TextEditingController _codeController = TextEditingController();

  bool _scanning = false;
  bool _connecting = false;
  bool _authenticating = false;
  String? _error;

  final List<BluetoothDevice> _hubs = [];
  BluetoothDevice? _selectedHub;

  // ---------------------------------------------------------------------------
  // Scan
  // ---------------------------------------------------------------------------

  Future<void> _startScan() async {
    setState(() {
      _scanning = true;
      _hubs.clear();
      _error = null;
    });

    try {
      await for (final device in _ble.scanForHubs()) {
        if (!mounted) return;
        setState(() => _hubs.add(device));
      }
    } catch (e) {
      if (mounted) setState(() => _error = 'Scan failed: $e');
    } finally {
      if (mounted) setState(() => _scanning = false);
    }
  }

  // ---------------------------------------------------------------------------
  // Connect
  // ---------------------------------------------------------------------------

  Future<void> _connectToHub(BluetoothDevice device) async {
    setState(() {
      _connecting = true;
      _error = null;
    });

    try {
      await _ble.connect(device);
      if (mounted) setState(() => _selectedHub = device);
    } catch (e) {
      if (mounted) setState(() => _error = 'Connection failed: $e');
    } finally {
      if (mounted) setState(() => _connecting = false);
    }
  }

  // ---------------------------------------------------------------------------
  // Authenticate
  // ---------------------------------------------------------------------------

  Future<void> _authenticate() async {
    final code = _codeController.text.trim();
    if (code.length != 12) {
      setState(() => _error = 'Code must be exactly 12 characters');
      return;
    }

    setState(() {
      _authenticating = true;
      _error = null;
    });

    try {
      final result = await _ble.authenticate(code);
      if (!mounted) return;

      switch (result) {
        case HubAuthResult.accepted:
          await _ble.subscribeStatusFeed();
          await _ble.subscribeMacList();
          if (mounted) context.go('/hub/exam');
        case HubAuthResult.rejected:
          setState(() => _error = 'Authentication rejected — check code');
        case HubAuthResult.timeout:
          setState(() => _error = 'Hub did not respond — try again');
      }
    } catch (e) {
      if (mounted) setState(() => _error = 'Auth error: $e');
    } finally {
      if (mounted) setState(() => _authenticating = false);
    }
  }

  // ---------------------------------------------------------------------------
  // Dispose
  // ---------------------------------------------------------------------------

  @override
  void dispose() {
    _codeController.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Hub Authentication')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: _selectedHub == null ? _buildScanView() : _buildAuthView(),
      ),
    );
  }

  // -- Scan & select hub ------------------------------------------------------

  Widget _buildScanView() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ElevatedButton.icon(
          onPressed: _scanning ? null : _startScan,
          icon: _scanning
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.bluetooth_searching),
          label: Text(_scanning ? 'Scanning...' : 'Scan for Hubs'),
        ),
        if (_error != null) ...[
          const SizedBox(height: 12),
          Text(_error!, style: const TextStyle(color: Colors.red)),
        ],
        const SizedBox(height: 16),
        Expanded(
          child: _hubs.isEmpty
              ? const Center(child: Text('No hubs found. Tap scan to begin.'))
              : ListView.separated(
                  itemCount: _hubs.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (context, index) {
                    final hub = _hubs[index];
                    return ListTile(
                      leading: const Icon(Icons.router),
                      title: Text(
                        hub.platformName.isNotEmpty
                            ? hub.platformName
                            : hub.remoteId.str,
                      ),
                      subtitle: Text(hub.remoteId.str),
                      trailing: _connecting
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child:
                                  CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.chevron_right),
                      onTap: _connecting ? null : () => _connectToHub(hub),
                    );
                  },
                ),
        ),
      ],
    );
  }

  // -- Auth code entry --------------------------------------------------------

  Widget _buildAuthView() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          'Connected to ${_selectedHub!.platformName.isNotEmpty ? _selectedHub!.platformName : _selectedHub!.remoteId.str}',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 32),
        TextField(
          controller: _codeController,
          maxLength: 12,
          textAlign: TextAlign.center,
          style: const TextStyle(
            fontSize: 24,
            letterSpacing: 4,
            fontFamily: 'monospace',
          ),
          inputFormatters: [
            FilteringTextInputFormatter.allow(RegExp(r'[A-Za-z0-9]')),
            UpperCaseTextFormatter(),
          ],
          decoration: const InputDecoration(
            labelText: 'Invigilator Code',
            hintText: 'XXXXXXXXXXXX',
            border: OutlineInputBorder(),
            counterText: '',
          ),
        ),
        const SizedBox(height: 24),
        ElevatedButton(
          onPressed: _authenticating ? null : _authenticate,
          child: _authenticating
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Text('Authenticate'),
        ),
        if (_error != null) ...[
          const SizedBox(height: 16),
          Text(
            _error!,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.red),
          ),
        ],
        const SizedBox(height: 16),
        TextButton(
          onPressed: () async {
            await _ble.disconnect();
            if (mounted) setState(() => _selectedHub = null);
          },
          child: const Text('Disconnect'),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Uppercase text formatter
// ---------------------------------------------------------------------------

class UpperCaseTextFormatter extends TextInputFormatter {
  @override
  TextEditingValue formatEditUpdate(
    TextEditingValue oldValue,
    TextEditingValue newValue,
  ) {
    return newValue.copyWith(
      text: newValue.text.toUpperCase(),
      selection: newValue.selection,
    );
  }
}
