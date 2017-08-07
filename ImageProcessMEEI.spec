# -*- mode: python -*-

block_cipher = None


a = Analysis(['ImageProcessMEEI.py'],
             pathex=['C:\\Users\\GUARIND\\Documents\\GitHub\\ImageProcessMEEI'],
             binaries=[],
             datas=[('C:\\Users\\GUARIND\\Documents\\GitHub\\ImageProcessMEEI\\shape_predictor_68_face_landmarks.dat','.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='ImageProcessMEEI',
          debug=False,
          strip=False,
          upx=True,
          console=False, icon='meei_3WR_icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='ImageProcessMEEI')
