import os
import gc
import json
from tqdm import tqdm
def get_pdb(pdb_code=""):

    # os.system(f"wget -qnc -P ones/ https://files.rcsb.org/view/{pdb_code}.pdb")
    os.system(f"wget -qnc -P all/ http://www.cathdb.info/version/v4_2_0/api/rest/id/{pdb_code}.pdb")
    return f"all/{pdb_code}.pdb"
## http://www.cathdb.info/version/v4_3_0/api/rest/id/<chain_id>.pdb

# pdb_code = '4frt.A'
# pdb_code = pdb_code[:4]
# get_pdb(pdb_code)
# exit()
from Bio.PDB import PDBParser, PDBIO,Select

# with open('chain_set_splits.json', 'r') as f:
#   data = json.load(f)

# pdb_name = []
# exits_file = os.listdir('all/')
# for key in data.keys():   # data.keys()
#     if key not in ['cath_nodes']:
#         for pdb_code in tqdm(data[key]):
#             pdb_code = pdb_code[:4]
#             if pdb_code+'.pdb' in exits_file:
#                 print(pdb_code,'exist')
#             else:
#                 get_pdb(pdb_code)
#                 print(pdb_code)

# err_file = []
# for key in ['./test', './train', './validation']:
#     if not os.path.exists(key):
#         os.makedirs(key)

class ChainSelector(Select):
    def accept_chain(self, chain):
        return chain.get_id() == chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "
    
# all_processed_file = os.listdir('test/') + os.listdir('train/')+os.listdir('validation/')
# for key in data.keys(): 
#     if key not in ['cath_nodes']:
#         for pdb_code in tqdm(data[key]):
#             if pdb_code+'.pdb' not in all_processed_file:
#                 pdb_file = f'all/{pdb_code[:4]}'+'.pdb'
#                 chain_id = pdb_code[5]

#                 parser = PDBParser(QUIET=True)
#                 try:
#                     print(pdb_file)
#                     structure = parser.get_structure(pdb_code[:4], pdb_file)

#                     io = PDBIO()

#                     io.set_structure(structure)
#                     io.save(key+f"/{pdb_code[:4]}.{chain_id}.pdb", ChainSelector())
#                     del io
#                     gc.collect()
#                 except FileNotFoundError:
#                     err_file.append(pdb_code)
# print(err_file)
    

pdb_file = f'/mnt/disks/wcls/Datas/origin_data/all/1vq8'+'.pdb'
chain_id = '3'

parser = PDBParser(QUIET=True)

print(pdb_file)
structure = parser.get_structure('4frt', pdb_file)

io = PDBIO()

io.set_structure(structure)
io.save('all'+f"/1vq8.{chain_id}.pdb", ChainSelector())
del io
gc.collect()