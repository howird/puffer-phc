{
  lib,
  pkgs,
  config,
  mkShell,
}:
mkShell rec {
  packages = with pkgs;
    [
      pixi
      glibc # pixi
      glib # pixi
      openssl_3
      libxcrypt-legacy # isaaclab
    ]
    ++ pythonManylinuxPackages.manylinux1
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

  PROJ_DIR = builtins.toString ./.;
  ENV_DIR = "${PROJ_DIR}/.pixi/envs/default";
  EXTRA_LIBRARIES = [
    "${ENV_DIR}"
    "${ENV_DIR}/lib/python3.8/site-packages/torch"
  ];

  env = lib.optionalAttrs pkgs.stdenv.isLinux {
    LD_LIBRARY_PATH = lib.makeLibraryPath (packages ++ EXTRA_LIBRARIES);
  };

  # shellHook = ''
  #   unset PYTHONPATH

  #   SOURCE_DATE_EPOCH=$(date +%s)
  #   if [ -d ".venv" ]; then
  #       echo "Skipping venv creation, '.venv' already exists"
  #   else
  #       echo "Creating new venv environment in path: '.venv'"
  #       uv venv
  #   fi
  #   source ".venv/bin/activate"
  #   uv sync --extra ${if config.cudaSupport then "cuda" else "cpu"}
  # '';
}
