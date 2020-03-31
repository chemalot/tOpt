#alberto



NumToName = \
    [ '*', 
      'H',                                           'He',
      'Li', 'Be', 'B',   'C',   'N',   'O',   'F',   'Ne',
      'Na', 'Mg', 'Al',  'Si',  'P',   'S',   'Cl',  'Ar',
    ]


NameToNum = { NumToName[n]: n for n in range(len(NumToName))}
