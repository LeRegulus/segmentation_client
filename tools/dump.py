import sys, nbformat
nb=nbformat.read(sys.argv[1],4)
for i,c in enumerate(nb.cells):
    if c.cell_type=='code':
        for o in c.outputs:
            if o.output_type=='stream': print(f'[{i}]',o.text[:1500])
            elif o.output_type=='error': print('ERR',o.ename,o.evalue)
            elif 'data' in o and 'text/plain' in o['data'] and 'image/png' not in o['data']: print(f'[{i}]',o['data']['text/plain'][:1500])
