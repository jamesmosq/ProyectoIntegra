"""
Script para ejecutar todas las pruebas del sistema predictor de deserción.

Uso:
    python ejecutar_pruebas.py            # Unitarias + Integradas
    python ejecutar_pruebas.py unitarias  # Solo pruebas unitarias
    python ejecutar_pruebas.py integradas # Solo pruebas integradas

Universidad Santo Tomás - Maestría en Ciencia de Datos
Autor: James Mosquera Rentería
"""

import sys
import io
import unittest

# Forzar UTF-8 en consolas Windows (cp1252 no soporta algunos caracteres unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def suite_unitarias():
    loader = unittest.TestLoader()
    return loader.loadTestsFromName("tests.test_unitarias")


def suite_integradas():
    loader = unittest.TestLoader()
    return loader.loadTestsFromName("tests.test_integradas")


def main():
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "todas"

    if modo == "unitarias":
        suite = suite_unitarias()
        titulo = "PRUEBAS UNITARIAS"
    elif modo == "integradas":
        suite = suite_integradas()
        titulo = "PRUEBAS DE INTEGRACIÓN"
    else:
        suite = unittest.TestSuite([suite_unitarias(), suite_integradas()])
        titulo = "PRUEBAS UNITARIAS + INTEGRADAS"

    print("=" * 65)
    print(f" {titulo}")
    print("=" * 65)

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    resultado = runner.run(suite)

    print("\n" + "=" * 65)
    print(f" RESUMEN: {resultado.testsRun} ejecutadas | "
          f"{len(resultado.failures)} fallidas | "
          f"{len(resultado.errors)} errores | "
          f"{resultado.testsRun - len(resultado.failures) - len(resultado.errors)} exitosas")
    print("=" * 65)

    sys.exit(0 if resultado.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
