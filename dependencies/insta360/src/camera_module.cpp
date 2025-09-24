#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <camera/camera.h>
#include <camera/photography_settings.h>
#include <camera/device_discovery.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace py = pybind11;

// Simple debug print function
#define DEBUG_LOG(msg) std::cout << "[Insta360Camera] " << msg << std::endl

class PyCameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> camera;
    std::shared_ptr<ins_camera::StreamDelegate> stream_delegate;
    
public:
    PyCameraWrapper() {
        DEBUG_LOG("Camera wrapper initialized");
    }
    
    bool discover_and_open() {
        DEBUG_LOG("Discovering devices...");
        ins_camera::DeviceDiscovery discovery;
        
        // Add more detailed debug info
        DEBUG_LOG("Checking USB permissions...");
        FILE* lsusb_output = popen("lsusb | grep -i insta360", "r");
        if (lsusb_output) {
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), lsusb_output)) {
                DEBUG_LOG("Found USB device: " << buffer);
            }
            pclose(lsusb_output);
        }
        
        try {
            DEBUG_LOG("Calling GetAvailableDevices()...");
            auto device_list = discovery.GetAvailableDevices();
            
            if (device_list.empty()) {
                DEBUG_LOG("No devices found by SDK. This might be a permissions issue.");
                DEBUG_LOG("Try: 1) Creating a udev rule or 2) Running with sudo");
                return false;
            }
            
            DEBUG_LOG("Found " << device_list.size() << " device(s). Opening first device...");
            
            try {
                DEBUG_LOG("Creating camera instance for device: " << device_list[0].serial_number);
                camera = std::make_shared<ins_camera::Camera>(device_list[0].info);
                
                DEBUG_LOG("Calling camera->Open()...");
                bool result = camera->Open();
                DEBUG_LOG("Camera open result: " << (result ? "success" : "failed"));
                discovery.FreeDeviceDescriptors(device_list);
                return result;
            } catch (const std::bad_alloc& e) {
                DEBUG_LOG("Memory allocation error when creating camera: " << e.what());
                DEBUG_LOG("This could be due to insufficient memory or a resource issue");
                DEBUG_LOG("Try: 1) Restart your application 2) Disconnect/reconnect camera 3) Check system resources");
                discovery.FreeDeviceDescriptors(device_list);
                return false;
            } catch (const std::exception& e) {
                DEBUG_LOG("Exception when creating camera: " << e.what());
                discovery.FreeDeviceDescriptors(device_list);
                return false;
            }
        } catch (const std::bad_alloc& e) {
            DEBUG_LOG("Memory allocation error during device discovery: " << e.what());
            DEBUG_LOG("This could be due to insufficient memory or a resource issue");
            DEBUG_LOG("Try: 1) Restart your application 2) Disconnect/reconnect camera 3) Check system resources");
            return false;
        } catch (const std::exception& e) {
            DEBUG_LOG("Exception during device discovery: " << e.what());
            return false;
        }
    }
    
    bool open_camera_by_index(int index) {
        DEBUG_LOG("Opening camera at index " << index);
        ins_camera::DeviceDiscovery discovery;
        auto device_list = discovery.GetAvailableDevices();
        
        if (device_list.empty() || index >= device_list.size()) {
            DEBUG_LOG("Invalid index or no devices found (available: " << device_list.size() << ")");
            return false;
        }
        
        DEBUG_LOG("Found device with serial: " << device_list[index].serial_number);
        camera = std::make_shared<ins_camera::Camera>(device_list[index].info);
        discovery.FreeDeviceDescriptors(device_list);
        
        bool result = camera->Open();
        DEBUG_LOG("Camera open result: " << (result ? "success" : "failed"));
        return result;
    }
    
    std::vector<std::string> get_available_devices() {
        DEBUG_LOG("Getting available devices...");
        ins_camera::DeviceDiscovery discovery;
        auto device_list = discovery.GetAvailableDevices();
        std::vector<std::string> result;
        
        DEBUG_LOG("Found " << device_list.size() << " device(s)");
        for (const auto& device : device_list) {
            result.push_back(device.serial_number);
            DEBUG_LOG("Device found: " << device.serial_number);
        }
        
        discovery.FreeDeviceDescriptors(device_list);
        return result;
    }
    
    bool start_recording(int resolution = static_cast<int>(ins_camera::VideoResolution::RES_3840_2160P30),
                         int bitrate = 30 * 1024 * 1024) {
        if (!camera) {
            DEBUG_LOG("Cannot start recording: Camera not initialized");
            return false;
        }
        
        DEBUG_LOG("Setting video mode to NORMAL");
        bool ret = camera->SetVideoSubMode(ins_camera::SubVideoMode::VIDEO_NORMAL);
        if (!ret) {
            DEBUG_LOG("Failed to set video mode");
            return false;
        }
        
        DEBUG_LOG("Configuring recording with resolution: " << resolution << ", bitrate: " << bitrate);
        ins_camera::RecordParams record_params;
        record_params.resolution = static_cast<ins_camera::VideoResolution>(resolution);
        record_params.bitrate = bitrate;
        
        if (!camera->SetVideoCaptureParams(record_params, ins_camera::CameraFunctionMode::FUNCTION_MODE_NORMAL_VIDEO)) {
            DEBUG_LOG("Failed to set video capture parameters");
            return false;
        }
        
        DEBUG_LOG("Starting recording...");
        bool result = camera->StartRecording();
        DEBUG_LOG("Start recording result: " << (result ? "success" : "failed"));
        return result;
    }
    
    std::vector<std::string> stop_recording() {
        if (!camera) {
            DEBUG_LOG("Cannot stop recording: Camera not initialized");
            return {};
        }
        
        DEBUG_LOG("Stopping recording...");
        auto url = camera->StopRecording();
        if (url.Empty()) {
            DEBUG_LOG("Stop recording returned empty URL");
            return {};
        }
        
        auto urls = url.OriginUrls();
        DEBUG_LOG("Recording stopped, got " << urls.size() << " main URLs");
        
        // Get all files currently on the camera
        auto all_camera_files = camera->GetCameraFilesList();
        
        std::vector<std::string> all_urls = urls;
        
        // For each main file, try to find its LRV counterpart
        for (const auto& main_url : urls) {
            DEBUG_LOG("Looking for LRV match for: " << main_url);
            
            // Parse the main URL to create the expected LRV pattern
            // Convert e.g., "/DCIM/Camera01/VID_20250206_095220_00_001.insv" 
            // to a pattern like "/DCIM/Camera01/LRV_20250206_095220_01_001.lrv"
            
            size_t last_slash = main_url.find_last_of('/');
            size_t underscore_pos = main_url.find("_00_");
            size_t dot_pos = main_url.find_last_of('.');
            
            if (last_slash != std::string::npos && 
                underscore_pos != std::string::npos && 
                dot_pos != std::string::npos) {
                
                std::string dir = main_url.substr(0, last_slash + 1);
                std::string pattern = main_url.substr(last_slash + 4, underscore_pos - last_slash - 4);
                std::string counter = main_url.substr(underscore_pos + 4, dot_pos - underscore_pos - 4);
                
                std::string lrv_pattern = dir + "LRV" + pattern + "_01_" + counter + ".lrv";
                DEBUG_LOG("Looking for LRV file matching pattern: " << lrv_pattern);
                
                // Search for matching LRV file in the camera files list
                for (const auto& file : all_camera_files) {
                    if (file.find(lrv_pattern) != std::string::npos) {
                        DEBUG_LOG("Found matching LRV file: " << file);
                        all_urls.push_back(file);
                        break;
                    }
                }
            }
        }
        
        return all_urls;
    }
    
    bool download_file(const std::string& camera_file_path, const std::string& local_file_path) {
        if (!camera) {
            DEBUG_LOG("Cannot download file: Camera not initialized");
            return false;
        }
        
        DEBUG_LOG("Downloading file from " << camera_file_path << " to " << local_file_path);
        
        // Simple progress callback
        ins_camera::DownloadProgressCallBack progress_callback = 
            [](int64_t download_size, int64_t total_size) {
                // Log progress at intervals
                static int last_percent = -1;
                int current_percent = (total_size > 0) ? (download_size * 100 / total_size) : 0;
                
                if (current_percent != last_percent && current_percent % 10 == 0) {
                    DEBUG_LOG("Download progress: " << download_size << "/" << total_size 
                              << " bytes (" << current_percent << "%)");
                    last_percent = current_percent;
                }
            };
        
        // Simple direct download
        try {
            bool result = camera->DownloadCameraFile(camera_file_path, local_file_path, progress_callback);
            DEBUG_LOG("Download result: " << (result ? "success" : "failed"));
            return result;
        } catch (const std::exception& e) {
            DEBUG_LOG("Exception during download: " << e.what());
            return false;
        } catch (...) {
            DEBUG_LOG("Unknown exception during download");
            return false;
        }
    }
    
    std::vector<std::string> get_camera_files_list() {
        if (!camera) {
            DEBUG_LOG("Cannot get files list: Camera not initialized");
            return {};
        }
        
        DEBUG_LOG("Retrieving camera files list...");
        auto files = camera->GetCameraFilesList();
        DEBUG_LOG("Found " << files.size() << " files on camera");
        return files;
    }
    
    std::string get_serial_number() {
        if (!camera) {
            DEBUG_LOG("Cannot get serial number: Camera not initialized");
            return "";
        }
        
        auto serial = camera->GetSerialNumber();
        DEBUG_LOG("Camera serial number: " << serial);
        return serial;
    }
    
    void close() {
        if (camera) {
            DEBUG_LOG("Closing camera connection");
            camera->Close();
        } else {
            DEBUG_LOG("Close called but camera was not initialized");
        }
    }
    
    ~PyCameraWrapper() {
        DEBUG_LOG("Camera wrapper being destroyed");
        close();
    }
};

PYBIND11_MODULE(_camera_sdk, m) {
    m.doc() = "Python bindings for the Insta360 Camera SDK";
    
    py::class_<PyCameraWrapper>(m, "Camera")
        .def(py::init<>())
        .def("discover_and_open", &PyCameraWrapper::discover_and_open, 
             "Discover available cameras and open the first one")
        .def("open_camera_by_index", &PyCameraWrapper::open_camera_by_index, 
             "Open a camera by its index in the available devices list", py::arg("index") = 0)
        .def("get_available_devices", &PyCameraWrapper::get_available_devices, 
             "Get a list of available camera devices by serial number")
        .def("start_recording", &PyCameraWrapper::start_recording, 
             "Start video recording", 
             py::arg("resolution") = static_cast<int>(ins_camera::VideoResolution::RES_2880_2880P60),
             py::arg("bitrate") = 10 * 1024 * 1024)
        .def("stop_recording", &PyCameraWrapper::stop_recording, 
             "Stop video recording and return both original and LRV file URLs")
        .def("download_file", &PyCameraWrapper::download_file, 
             "Download a file from the camera", 
             py::arg("camera_file_path"), py::arg("local_file_path"))
        .def("get_camera_files_list", &PyCameraWrapper::get_camera_files_list, 
             "Get a list of files on the camera")
        .def("get_serial_number", &PyCameraWrapper::get_serial_number, 
             "Get the camera serial number")
        .def("close", &PyCameraWrapper::close,
             "Close the camera connection");
    
    // Enum for video resolutions
    py::enum_<ins_camera::VideoResolution>(m, "VideoResolution")
        .value("RES_2880_2880P60", ins_camera::VideoResolution::RES_2880_2880P60)
        .value("RES_2880_2880P30", ins_camera::VideoResolution::RES_2880_2880P30)
        .value("RES_5120_5120P30", ins_camera::VideoResolution::RES_5120_5120P30)
        .export_values();
}