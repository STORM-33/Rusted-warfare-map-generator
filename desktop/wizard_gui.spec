# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['scipy._cyutility']
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')


a = Analysis(
    ['wizard_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('generator_blueprint1.tmx', '.'), ('generator_blueprint2.tmx', '.'), ('generator_blueprint3.tmx', '.'), ('generator_blueprint4.tmx', '.'), ('generator_blueprint5.tmx', '.')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rthook_stdio.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='wizard_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
