## Master thesis
## Data
The data was obtained from PPI3D web server in 2024 spring (!could be updated) https://bioinformatics.lt/ppi3d/clusters selecting all protein-nucleic acid interations (both DNA and RNA) with lower that 2.5A resolution. To get rid of ribosomes the nucleic acid chain lenght upper boundary is set to 150 nucleotides and lower boundary to 8 nucleotides, to remove tiny fragments of nucleic acids. Protein-NA complexes are separeted into 3 groups based on nucleic acid: dsDNA, ssDNA, RNA. 
Lots of filtering and manual observations were to to separate these 3 groups based on PPI3D annotations. 
