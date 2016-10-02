universal_tagset = {
    'a' : 'ADJ',
    'r' : 'ADV',
    'd' : 'DET',
    'v' : 'VB',
    'p' :  'PRON',
    'c' : 'CC',
    'i' : 'CC',
    's' :  'PREP',
    'z' :  'NUM',
}

def simplify_tagset(pos):
    if not pos:
        return pos
    new_pos = pos
    if pos[0]=='n':
        new_pos=pos[:2].upper()
    if pos[0] in universal_tagset:
        new_pos = universal_tagset[pos[0]]
    if pos[0] == 'f':   # puntuacion
        if pos[:2] == 'fa' or pos[:2]=='fi':
            new_pos = 'fa'
        if (pos[:2] == 'fc' or pos[:2] == 'fl') and len(pos) > 2:
            new_pos = 'fl'
        if pos == 'fc' or pos == 'fx':
            new_pos = 'fc'
        if pos == 'fe' or pos[:2] == 'fr':
            new_pos = 'fe'
    return new_pos
