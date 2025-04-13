{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";

    nixpkgs-system.url = "github:NixOS/nixpkgs/c8cd81426f45942bb2906d5ed2fe21d2f19d95b7";
    # nixgl = {
    #   url = "github:nix-community/nixGL";
    #   inputs.nixpkgs.follows = "nixpkgs";
    #   inputs.flake-utils.follows = "utils";
    # };
  };

  outputs = {
    nixpkgs,
    nixpkgs-system,
    # nixgl,
    ...
  } @ inputs:
    inputs.utils.lib.eachSystem ["x86_64-linux"] (system: let
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
      pkgs = import nixpkgs {
        inherit system config;
      };
      pkgs-system = import nixpkgs-system {
        inherit system config;
      };
    in {
      devShells = {
        default = pkgs.callPackage ./nix/dev-shells/uv-impure.nix {
          inherit pkgs-system;
        };
      };

      formatter = nixpkgs.legacyPackages.${system}.alejandra;
    });
}
