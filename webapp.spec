# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 — RAG 知识库系统
构建: pyinstaller webapp.spec
输出: dist/RAG知识库系统.exe
"""

a = Analysis(
    ['webapp.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates/index.html', 'templates'),
    ],
    hiddenimports=[
        'chromadb',
        'chromadb.config',
        'chromadb.api',
        'chromadb.utils.embedding_functions',
        'langchain',
        'langchain_core',
        'langchain_openai',
        'langchain_community',
        'langchain_community.document_loaders',
        'langchain_community.document_loaders.pdf',
        'rank_bm25',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'fastapi',
        'pydantic',
        'dotenv',
        'sqlite3',
        'json',
        'hashlib',
        'uuid',
        're',
        'pathlib',
        'numpy',
        'onnxruntime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'transformers',
        'sentence_transformers',
        'tensorflow',
        'tensorboard',
        'scipy',
        'pandas',
        'matplotlib',
        'PIL',
        'cv2',
        'sklearn',
    ],
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
    name='RAG知识库系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
