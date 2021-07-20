# %%
# grab the AHBA genes for standard MNI space
import abagen

gene_path = r'G:\006pd_DTI\05_gene_analysis\01_AHBAexpression\expression.csv'

expression = abagen.get_expression_data('./brainnetome.nii')
expression.to_csv(gene_path)