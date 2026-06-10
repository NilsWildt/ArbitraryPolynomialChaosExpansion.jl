# BLISBLASExt.jl - Extension for BLISBLAS support
# This extension is only loaded when BLISBLAS is available

module BLISBLASExt

using BLISBLAS

# BLISBLAS is automatically loaded when this extension is triggered
# No additional setup needed - the package provides optimized BLAS operations

end # module
