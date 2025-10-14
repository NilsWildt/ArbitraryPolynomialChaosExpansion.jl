{
  description = "ArbitraryPolynomialChaosExpansion.jl - A Julia package for arbitrary polynomial chaos expansion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
  };

  outputs = { self, nixpkgs, flake-utils, claude-code }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Create development environment for the main project
        devEnv = pkgs.mkShell {
          name = "apce-dev";
          
          buildInputs = with pkgs; [
            # Additional development tools
            git
            gcc
            gfortran
            # For CairoMakie plotting
            cairo
            pango
            gdk-pixbuf
          ] ++ [
            # Claude Code from external flake
            claude-code.packages.${system}.default
          ];
          
          shellHook = ''
            echo "ArbitraryPolynomialChaosExpansion.jl development environment loaded"
            echo "Using global Julia installation"
            echo ""
            echo "Available commands:"
            echo "  julia --project=. -e 'using Pkg; Pkg.instantiate()'  # Install dependencies"
            echo "  julia --project=. -e 'using APCE'                     # Test import"
            echo "  julia --project=. test/runtests.jl                    # Run tests"
            echo "  claude-code                                           # Start Claude Code"
          '';
          
          # Environment variables for Julia
          JULIA_DEPOT_PATH = "~/.julia";
          JULIA_NUM_THREADS = "auto";
          
          # For CairoMakie
          GDK_PIXBUF_MODULE_FILE = "${pkgs.librsvg.out}/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache";
        };
        
      in {
        # Default development environment
        devShells.default = devEnv;
        
        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
