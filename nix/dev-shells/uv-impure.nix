{
  lib,
  pkgs,
  config,
  mkShell,
  python310,
}:
mkShell rec {
  packages = with pkgs;
    [
      uv
      libxcrypt-legacy # isaaclab
    ]
    ++ (import ../pkgsets/gl.nix pkgs)
    ++ lib.optionals config.cudaSupport (with pkgs.cudaPackages; [
      cuda_cudart
      # cuda_nvcc
      cuda_cupti

      cudnn
      nccl
      libcusparse
      libcurand
      libcusolver
      libcufft
      libcublas

      pkgs.linuxPackages.nvidia_x11_vulkan_beta
    ]);

  env =
    {
      UV_PYTHON_DOWNLOADS = "never";
      UV_PYTHON = python310.interpreter;
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux {
      LD_LIBRARY_PATH = lib.makeLibraryPath (pkgs.pythonManylinuxPackages.manylinux1 ++ packages);
    };

  shellHook = ''
    unset PYTHONPATH

    SOURCE_DATE_EPOCH=$(date +%s)
    if [ -d ".venv" ]; then
        echo "Skipping venv creation, '.venv' already exists"
    else
        echo "Creating new venv environment in path: '.venv'"
        uv venv
    fi
    source ".venv/bin/activate"
    uv sync --extra ${if config.cudaSupport then "cuda" else "cpu"}
  '';
}
