import requests
import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import librosa.display
import streamlit as st
import numpy as np
import soundfile as sf
from datetime import datetime


# Função para baixar um arquivo de áudio
def download_audio(url, output_file):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        return output_file
    else:
        st.error(f"Erro ao baixar o áudio: {url}")
        return None


# Função para gerar um sonograma
def generate_sonogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis', ax=ax)
    ax.set(title=f'Sonograma: {os.path.basename(audio_file)}', xlabel='Tempo (s)', ylabel='Frequência (Hz)')
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    return fig


# Função para amplificar o som
def amplify_to_max(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    max_amplitude = np.max(np.abs(y))
    amplification_factor = 0.99 / max_amplitude if max_amplitude > 0 else 1.0
    y_amplified = y * amplification_factor
    amplified_file = "amplified_" + os.path.basename(audio_file)
    sf.write(amplified_file, y_amplified, sr)
    return amplified_file


# Função para identificar espécies inéditas após uma data
def get_new_species(df, reference_date):
    df_copy = df.copy()

    try:
        df_copy['date_only'] = df_copy['Timestamp'].str[:10]
        df_copy['date_only'] = pd.to_datetime(df_copy['date_only'])
        reference_date = pd.to_datetime(reference_date).date()
        
        before_date = df_copy[df_copy['date_only'].dt.date < reference_date]
        after_date = df_copy[df_copy['date_only'].dt.date >= reference_date]
        
        species_before = set(before_date['Scientific Name'].unique())
        species_after = set(after_date['Scientific Name'].unique())
        
        new_species = species_after - species_before
        return list(new_species)
    except Exception as e:
        st.error(f"Erro ao processar datas: {str(e)}")
        return []


# Função para gerar log de exclusões
def save_deletion_log():
    if not st.session_state.deletion_log:
        st.warning("Nenhuma exclusão foi registrada nesta sessão.")
        return None, None
        
    log_content = "=== Log de Exclusões ===\n\n"
    log_content += f"Arquivo original: {st.session_state.original_filename}\n\n"
    log_content += "Espécies excluídas:\n"
    
    for entry in st.session_state.deletion_log:
        log_content += f"\n- {entry['timestamp']}"
        log_content += f"\nEspécie: {entry['species']}"
        log_content += f"\nRegistros removidos: {entry['count']}\n"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"deletion_log_{timestamp}.txt"
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    return log_filename, log_content


# Inicializar variáveis no session_state
if 'new_species_list' not in st.session_state:
    st.session_state.new_species_list = []

if 'reference_date' not in st.session_state:
    st.session_state.reference_date = None

if 'excluded_species' not in st.session_state:
    st.session_state.excluded_species = set()

if 'deletion_log' not in st.session_state:
    st.session_state.deletion_log = []
    
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None


# Dashboard com Streamlit
st.title("Dashboard de Sonogramas e Edição de Tabela")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregue a tabela com as URLs dos arquivos de áudio", type=['csv'])

if uploaded_file:
    try:
        # Atualizar nome do arquivo original e resetar log se necessário
        if st.session_state.original_filename != uploaded_file.name:
            st.session_state.original_filename = uploaded_file.name
            st.session_state.deletion_log = []

        # Ler o arquivo CSV
        df = pd.read_csv(uploaded_file)

        if 'Soundscape' not in df.columns or 'Scientific Name' not in df.columns:
            st.error("A tabela deve conter as colunas 'Soundscape' e 'Scientific Name'.")
        else:
            # Adicionar sidebar para filtros
            st.sidebar.title("Filtros")

            # Filtros de Score/Confidence/Probability
            st.sidebar.subheader("Filtros de Qualidade")

            df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
            df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
            df['Probability'] = pd.to_numeric(df['Probability'], errors='coerce')

            min_score = st.sidebar.slider("Score Mínimo",
                                          min_value=float(df['Score'].min()),
                                          max_value=float(df['Score'].max()),
                                          value=float(df['Score'].min()))

            min_confidence = st.sidebar.slider("Confidence Mínima",
                                               min_value=float(df['Confidence'].min()),
                                               max_value=float(df['Confidence'].max()),
                                               value=float(df['Confidence'].min()))

            min_probability = st.sidebar.slider("Probability Mínima",
                                                min_value=float(df['Probability'].min()),
                                                max_value=float(df['Probability'].max()),
                                                value=float(df['Probability'].min()))

            # Aplicar filtros de qualidade
            df_filtered = df[
                (df['Score'] >= min_score) &
                (df['Confidence'] >= min_confidence) &
                (df['Probability'] >= min_probability)
            ]

            # Aplicar filtro de espécies excluídas
            if st.session_state.excluded_species:
                df_filtered = df_filtered[~df_filtered['Scientific Name'].isin(st.session_state.excluded_species)]

            # Filtro de espécies inéditas
            st.sidebar.subheader("Descobrir Espécies Inéditas")

            df['date_only'] = df['Timestamp'].str[:10]
            df['date_only'] = pd.to_datetime(df['date_only'])
            min_date = df['date_only'].min().date()
            max_date = df['date_only'].max().date()

            if 'reference_date' not in st.session_state:
                st.session_state.reference_date = min_date

            def update_new_species():
                st.session_state.new_species_list = get_new_species(df, st.session_state.reference_date)

            # Botões para seleção rápida de datas
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Último dia"):
                st.session_state.reference_date = max_date - pd.Timedelta(days=1)
                update_new_species()
            if col2.button("Últimos 3 dias"):
                st.session_state.reference_date = max_date - pd.Timedelta(days=3)
                update_new_species()

            col3, col4 = st.sidebar.columns(2)
            if col3.button("Última semana"):
                st.session_state.reference_date = max_date - pd.Timedelta(days=7)
                update_new_species()
            if col4.button("Último mês"):
                st.session_state.reference_date = max_date - pd.Timedelta(days=30)
                update_new_species()

            reference_date = st.sidebar.date_input(
                "Ou selecione uma data específica",
                value=st.session_state.reference_date,
                min_value=min_date,
                max_value=max_date
            )

            if st.sidebar.button("Buscar Espécies Inéditas"):
                st.session_state.new_species_list = get_new_species(df, reference_date)
                st.session_state.reference_date = reference_date

            if st.session_state.new_species_list:
                st.sidebar.success(
                    f"Encontradas {len(st.session_state.new_species_list)} espécies inéditas após {st.session_state.reference_date}")
                st.sidebar.write("Espécies inéditas:")
                for species in st.session_state.new_species_list:
                    st.sidebar.write(f"- {species}")
            elif st.session_state.reference_date:
                st.sidebar.info("Nenhuma espécie inédita encontrada após a data selecionada")

            # Adicionar seção de log na sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("Registro de Exclusões")
            
            if st.sidebar.button("Gerar arquivo de log"):
                log_file, log_content = save_deletion_log()
                if log_file and log_content:
                    st.sidebar.download_button(
                        "Baixar arquivo de log",
                        log_content,
                        file_name=log_file,
                        mime="text/plain"
                    )
                    st.sidebar.success(f"Log gerado com sucesso!")

            # Mostrar resumo das exclusões
            if st.session_state.deletion_log:
                st.sidebar.markdown("### Resumo das exclusões:")
                total_species = len(set(entry['species'] for entry in st.session_state.deletion_log))
                total_records = sum(entry['count'] for entry in st.session_state.deletion_log)
                st.sidebar.write(f"Total de espécies excluídas: {total_species}")
                st.sidebar.write(f"Total de registros removidos: {total_records}")

            # Calcular species_counts após aplicar todos os filtros
            species_counts = df_filtered['Scientific Name'].value_counts(ascending=True)
            sorted_species = species_counts.index.tolist()

            # Seleção de espécies para visualização
            selected_species = st.multiselect(
                "Selecione as espécies para filtrar:",
                sorted_species,
                format_func=lambda x: f"{x} ({species_counts[x]} registros)"
            )

            if selected_species:
                temp_filtered_df = df_filtered[df_filtered['Scientific Name'].isin(selected_species)]

                st.write("Gerando sonogramas para até 50 arquivos das espécies selecionadas...")

                cols = st.columns(4)

                for i, (index, row) in enumerate(temp_filtered_df.head(50).iterrows()):
                    with cols[i % 4]:
                        st.write(f"Arquivo {i + 1}:")
                        st.write(f"**Espécie:** {row['Scientific Name']}")
                        st.write(f"**Data:** {row['Timestamp']}")

                        temp_audio_file = f"temp_audio_{i + 1}.wav"
                        audio_file = download_audio(row['Soundscape'], temp_audio_file)

                        if audio_file:
                            amplified_file = amplify_to_max(audio_file)
                            fig = generate_sonogram(audio_file)
                            st.pyplot(fig)
                            st.audio(amplified_file, format='audio/wav')
                            os.remove(audio_file)
                            os.remove(amplified_file)

                st.success("Processamento concluído!")

            # Edição da tabela
            st.write("### Edição da Tabela")
            available_species = sorted_species  # Já está filtrado corretamente
            species_to_delete = st.multiselect(
                "Selecione as espécies que deseja excluir da tabela:",
                available_species
            )

            if st.button("Apagar espécies") and species_to_delete:
                for species in species_to_delete:
                    count_before = len(df_filtered)
                    df_filtered = df_filtered[~df_filtered['Scientific Name'].isin([species])]
                    records_removed = count_before - len(df_filtered)
                    
                    log_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'species': species,
                        'count': records_removed
                    }
                    st.session_state.deletion_log.append(log_entry)
                
                st.session_state.excluded_species.update(species_to_delete)
                st.success(f"Excluídos registros das espécies: {', '.join(species_to_delete)}")

            # Exibir e baixar a nova tabela
            st.write("### Tabela Atualizada")
            st.dataframe(df_filtered)
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Baixar tabela atualizada",
                data=csv,
                file_name="tabela_atualizada.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")