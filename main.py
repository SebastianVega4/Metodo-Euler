import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO

# Configuración inicial de la página
st.set_page_config(page_title="Método de Euler Mejorado", page_icon="📈", layout="wide")

# Verificar e instalar dependencias faltantes
try:
    import openpyxl
except ImportError:
    st.warning("La librería openpyxl no está instalada. Instalándola ahora...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    import openpyxl

# Configuración de matplotlib para Streamlit
plt.switch_backend('agg')  # Necesario para evitar problemas con Streamlit

# Métodos numéricos
def euler_method(x, y_initial, dydx_func, dx):
    """Implementación del método de Euler."""
    y = np.zeros(len(x))
    y[0] = y_initial
    for i in range(len(x)-1):
        y[i+1] = y[i] + dydx_func(x[i], y[i]) * dx
    return y

def exact_solution(x, func):
    """Calcula la solución exacta si se proporciona la función."""
    try:
        return func(x)
    except Exception as e:
        st.warning(f"Error al calcular solución exacta: {str(e)}")
        return None

def create_results_table(x, y_euler, y_exact=None):
    """Crea un DataFrame con los resultados."""
    data = {'x': x, 'y_euler': y_euler}
    if y_exact is not None:
        data['y_exact'] = y_exact
        data['error'] = np.abs(y_exact - y_euler)
    return pd.DataFrame(data)

def plot_solutions(x, y_euler, y_exact=None):
    """Genera el gráfico de comparación de soluciones."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Configuración del gráfico
    ax.plot(x, y_euler, 'bo-', linewidth=2, markersize=6, label='Aproximación (Euler)')
    if y_exact is not None:
        ax.plot(x, y_exact, 'r-', linewidth=2, label='Solución Exacta')
    
    ax.set_title("Comparación de Soluciones", fontsize=14, pad=20)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)
    
    # Mejorar los márgenes
    plt.tight_layout()
    
    return fig

def plot_errors(x, error):
    """Genera el gráfico de errores."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, error, 'm-', linewidth=2, label='Error absoluto')
    ax.set_title("Error de Aproximación", fontsize=14, pad=20)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Mejorar los márgenes
    plt.tight_layout()
    
    return fig

# Interfaz de usuario
def user_interface():
    """Configura la interfaz de usuario con Streamlit."""
    
    # Barra lateral para parámetros de entrada
    with st.sidebar:
        st.header("⚙️ Parámetros de Entrada")
        
        st.subheader("Ecuación Diferencial By. Sebastian Vega")
        st.markdown("Ingrese la derivada dy/dx = f(x, y) en formato matemarico")
        derivative_input = st.text_input("dy/dx =", "2*x-3*y+1")
        
        st.subheader("Condición Inicial")
        x0 = st.number_input("x inicial (x₀)", value=1.0)
        y0 = st.number_input("y inicial (y₀)", value=5.0)
        
        st.subheader("Rango de Solución")
        x_start = st.number_input("Inicio del intervalo", value=1.0)
        x_end = st.number_input("Fin del intervalo", value=1.5)
        step_size = st.number_input("Tamaño de paso (h)", min_value=0.001, value=0.05, step=0.01, format="%.3f")
        
        st.subheader("Opciones Adicionales")
        show_exact = st.checkbox("Mostrar solución exacta", value=False)
        if show_exact:
            exact_solution_input = st.text_input("Solución exacta y =", "(2 * x - 1/9) + math.exp(-3 * x)")
        
        calculate_button = st.button("Calcular Solución", type="primary", use_container_width=True)
    
    # Contenido principal
    st.title("📈 Método de Euler para EDOs " \
    "")
    st.markdown("""
    Esta aplicación implementa el **método de Euler** para resolver ecuaciones diferenciales ordinarias (EDOs) 
    de primer orden con valores iniciales (problemas de Cauchy).
    """)
    
    if calculate_button:
        try:
            # Validación de entradas
            if x_end <= x_start:
                st.error("El valor final debe ser mayor que el inicial")
                return
            if step_size <= 0:
                st.error("El tamaño de paso debe ser positivo")
                return
            
            # Crear el rango x
            num_points = int((x_end - x_start) / step_size) + 1
            x = np.linspace(x_start, x_end, num_points)
            
            # Definir la función derivada
            try:
                def dydx(x_val, y_val):
                    return eval(derivative_input, {'x': x_val, 'y': y_val, 'math': math, 'np': np})
            except Exception as e:
                st.error(f"Error al procesar la derivada: {str(e)}")
                st.error("Ejemplo de formato válido: '2*x' o 'y - x'")
                return
            
            # Calcular solución con Euler
            y_euler = euler_method(x, y0, dydx, step_size)
            
            # Calcular solución exacta si se proporciona
            y_exact = None
            if show_exact:
                try:
                    def exact_func(x_val):
                        return eval(exact_solution_input, {'x': x_val, 'math': math, 'np': np})
                    y_exact = exact_solution(x, exact_func)
                except Exception as e:
                    st.warning(f"No se pudo calcular la solución exacta: {str(e)}")
                    show_exact = False
            
            # Crear tabla de resultados
            df = create_results_table(x, y_euler, y_exact)
            
            # Mostrar resultados
            with st.expander("📊 Resultados Numéricos", expanded=True):
                st.dataframe(df.style.format("{:.6f}"), use_container_width=True)
                
                # Opciones de exportación
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Descargar CSV",
                        data=csv,
                        file_name="resultados_euler.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    try:
                        excel_buffer = BytesIO()
                        df.to_excel(excel_buffer, index=False, engine='openpyxl')
                        st.download_button(
                            "Descargar Excel",
                            data=excel_buffer,
                            file_name="resultados_euler.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"No se pudo generar el archivo Excel: {str(e)}")
            
            # Mostrar gráfico
            with st.expander("📈 Gráfico de Resultados", expanded=True):
                fig = plot_solutions(x, y_euler, y_exact)
                st.pyplot(fig, clear_figure=True)  # clear_figure ayuda a liberar memoria
                
                # Opción para descargar el gráfico
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                st.download_button(
                    "Descargar Gráfico",
                    data=buf,
                    file_name="grafico_euler.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # Análisis de error
            if show_exact and y_exact is not None:
                with st.expander("📉 Análisis de Error", expanded=False):
                    error = np.abs(y_exact - y_euler)
                    max_error = np.max(error)
                    mean_error = np.mean(error)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Error máximo", f"{max_error:.6f}")
                    with col2:
                        st.metric("Error promedio", f"{mean_error:.6f}")
                    
                    fig_err = plot_errors(x, error)
                    st.pyplot(fig_err, clear_figure=True)
        
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {str(e)}")
            st.error("Por favor verifica tus entradas y vuelve a intentar.")

# Ejecutar la aplicación
if __name__ == "__main__":
    user_interface()