# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['NJmat.py' ,
     '.\\NJmatML\\dataML.py',
     '.\\Visualizer\\ase_gui.py',
     '.\\MP\\CIF download.py' , 
     '.\\MP\\Descriptor design.py'
     ],
     
    pathex=[],
    binaries=[],
    datas=[
        (".\\CSP", "CSP"), 
        (".\\Visualizer", "Visualizer"),  
        (".\\MP", "MP")
    ],
    hiddenimports=[
        'ase',
        'BeautifulReport',  
        'catboost',
        'chemdataextractor',
        'dpdata',
        'dscribe',
        'gensim',
        'ghapi', 
        'gplearn',
        'graphviz',
        'ipython',  
        'matminer',
        'mendeleev',
        'mpi4py',
        'packaging',
        'padelpy',
        'paramiko',
        'plotly',
        'pydotplus',
        'pymatgen',
        'PyQt5',
        'PyQt5-sip',
        'quippy',
        'rdkit',
        'scikit-learn',
        'scipy',
        'seaborn',
        'spglib',
        'tensorflow',
        'xgboost'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NJmat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NJmat',
)
