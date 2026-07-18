{ pkgs ? import <nixpkgs> {
    config = {
        allowUnfree = true;
    }; 
  }
}:

pkgs.mkShell.override { stdenv = pkgs.gccStdenv; } {
  # Tools required during compilation
  nativeBuildInputs = with pkgs; [
    gcc
    gnumake
    pkg-config
    cmake
    uv  # Added uv so anyone using this shell has it installed
    pyright
  ];

  # Development libraries and headers
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
    
    # --- ADDED FOR UV / PYTHON COMPATIBILITY ---
    libxcrypt-legacy  # Provides libcrypt.so.1 for pre-compiled Python binaries
  ];

  # Required to bypass static linking issues
  GLIBC_TUNABLES = "glibc.rtld.execstack=2";

  shellHook = ''
    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"

    # required for python interpreter to work
    export NIX_LD="${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2"
    
    # 1. Handle system-level OpenGL drivers (NixOS specific paths)
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
    
    # 2. Handle Nix-provided libraries using makeLibraryPath for cleaner syntax
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
      stdenv.cc.cc.lib
      libGL
      libglvnd
      mesa
      libxcrypt-legacy
      zlib
    ])}:$LD_LIBRARY_PATH"

    # required for python interpreter to work
    export NIX_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    
    # 3. Automatically handle the python virtual environment
    if [ ! -d .venv ]; then
      echo "No virtual environment found. Running uv sync..."
      uv sync
    fi
    
    source .venv/bin/activate

    echo "Development environment loaded! Compilers and Python are ready."
  '';
}