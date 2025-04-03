{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";

    nixpkgs-stable.url = "github:NixOS/nixpkgs/release-24.11";
    # nixgl = {
    #   url = "github:nix-community/nixGL";
    #   inputs.nixpkgs.follows = "nixpkgs";
    #   inputs.flake-utils.follows = "utils";
    # };
  };

  outputs = {
    nixpkgs,
    nixpkgs-stable,
    # nixgl,
    ...
  } @ inputs:
    inputs.utils.lib.eachSystem ["x86_64-linux"] (system: let
      config = {
        allowUnfree = true;
        # cudaSupport = true;
      };
      pkgs = import nixpkgs {
        inherit system config;
      };
      pkgs-stable = import nixpkgs-stable {
        inherit system config;
      };
    in {
      devShells = rec {
        default = uv-impure;
        pixi-impure = pkgs.callPackage ./nix/dev-shells/pixi-impure.nix {};
        uv-impure = pkgs.callPackage ./nix/dev-shells/uv-impure.nix {};
        fhsenv = pkgs.callPackage ./nix/dev-shells/fhsenv.nix {};
      };

      formatter = nixpkgs.legacyPackages.${system}.alejandra;
    });
}
