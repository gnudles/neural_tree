import 'dart:convert';
import 'dart:io';

/// Saves the network to file.
Future<void> saveJson(String filename, Map<String, dynamic> map) async {
  var jsonString = jsonEncode(map);
  jsonString.replaceAll("}", "}\n");
  var sink = File(filename).openWrite(mode: FileMode.writeOnly);
  sink.write(jsonString);

  // Close the IOSink to free system resources.
  await sink.flush();
  sink.close();
}

Map<String, dynamic>? loadJson(String filename) {
  try {
    String jsonString = File(filename).readAsStringSync();
    return jsonDecode(jsonString);
  } catch (e) {
    return null;
  }
}
