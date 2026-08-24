{ pkgs ? import <nixpkgs> {
    config = {
        allowUnfree = true;
        cudaSupport = true; 
    }; 
  }
}:

pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
  nativeBuildInputs = with pkgs; [
    gcc
    gnumake
    pkg-config
    cmake
    uv  
    pyright
  ];

  buildInputs = with pkgs; [
    glibc.dev
    zlib
    libGL
    libGLU
    mesa
    pkg-config
    cacert
    patchelf
    stdenv.cc.cc.lib
    alsa-lib
    vulkan-loader
    libxcrypt-legacy  

    # --- ADDED FOR NVIDIA / TENSORRT SUPPORT ---
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
  ];

  GLIBC_TUNABLES = "glibc.rtld.execstack=2";

  shellHook = ''
    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
    export NIX_LD="${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2"
    
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
    
    # Inject CUDA and TensorRT directly into the LD path
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
      stdenv.cc.cc.lib
      libGL
      libglvnd
      mesa
      libxcrypt-legacy
      zlib
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      cudaPackages.tensorrt
    ])}:$LD_LIBRARY_PATH"

    export NIX_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    
    if [ ! -d .venv ]; then
      echo "No virtual environment found. Running uv sync..."
      uv sync
    fi
    
    source .venv/bin/activate
    echo "Development environment loaded! NVIDIA TensorRT is ready."
  '';
}