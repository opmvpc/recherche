@echo off
echo ============================================
echo Nettoyage COMPLET du cache Streamlit
echo ============================================
echo.

echo [1/4] Suppression __pycache__...
if exist __pycache__ rmdir /s /q __pycache__
if exist src\__pycache__ rmdir /s /q src\__pycache__
echo OK

echo [2/4] Suppression .streamlit\cache...
if exist .streamlit\cache rmdir /s /q .streamlit\cache
echo OK

echo [3/4] Suppression cache utilisateur Streamlit...
if exist "%USERPROFILE%\.streamlit\cache" rmdir /s /q "%USERPROFILE%\.streamlit\cache"
echo OK

echo [4/4] Suppression data/cache...
if exist data\cache rmdir /s /q data\cache
echo OK

echo.
echo ============================================
echo Cache vide! Relance Streamlit maintenant.
echo ============================================
echo.
echo Commande: streamlit run app.py
pause

