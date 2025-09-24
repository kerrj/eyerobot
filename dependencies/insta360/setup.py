import os
import sys
import platform
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Get the path to the dependencies directory relative to setup.py
DEPENDENCIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dependencies")

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Make sure pybind11 is available
        self._ensure_pybind11()

        # Set Python_EXECUTABLE for CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # Add pybind11 directory if it exists in the current directory
            f"-Dpybind11_DIR={os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pybind11')}",
        ]
        
        # Determine the platform-specific library paths
        if platform.system() == "Windows":
            lib_ext = "lib"
        elif platform.system() == "Darwin":
            lib_ext = "dylib"
        else:
            lib_ext = "so"
            
        # Set Camera SDK paths
        camera_sdk_include = os.path.join(DEPENDENCIES_DIR, "CameraSDK", "include")
        camera_sdk_lib = os.path.join(DEPENDENCIES_DIR, "CameraSDK", "lib")
        
        cmake_args.extend([
            f"-DCAMERA_SDK_INCLUDE_DIR={camera_sdk_include}",
            f"-DCAMERA_SDK_LIB_DIR={camera_sdk_lib}",
        ])

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja
                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})
            
            if not single_config and not contains_arch:
                cmake_args += ["-A", "x64"]
            
            build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

    def _ensure_pybind11(self):
        """Make sure pybind11 is available"""
        pybind11_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pybind11')
        
        # If pybind11 directory doesn't exist, clone it
        if not os.path.exists(pybind11_dir):
            subprocess.check_call(
                ["git", "clone", "https://github.com/pybind/pybind11.git", pybind11_dir]
            )

setup(
    name="insta360",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python bindings for the Insta360 Camera SDK",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/insta360-python",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("insta360._camera_sdk")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "pybind11>=2.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)