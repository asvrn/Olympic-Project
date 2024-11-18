import streamlit as st
from modules.input import load_dataset
from modules.preprocessing import detailed_eda
from modules.analysis import perform_analysis
from modules.visualization import create_visualization

def show_home():
    # Judul Utama
    st.title("🏅Dashboard Olympic Games 1896-2022")
    st.markdown("---")

    # Informasi Kelompok Project
    st.header("👥 Project Group 6 Data Aquisition-B")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anggota Tim:")
        st.markdown("""
        * Muhammad Harsya (2211521005)
        * Vioni Wijaya Putri (2211522016)
        * Rifqi Asverian Putra (2211522021)
        * Miftahul Jannah (2211522024)
        """)
    
    with col2:
        st.subheader("Dosen Pembimbing:")
        st.markdown(" Rahmatika Pratama Santi, MT ")

    st.markdown("---")

    # Informasi tentang Website
    st.header("💻 Tentang Website")
    st.write("""
    Website ini merupakan platform analisis data yang fokus pada pengolahan dan visualisasi
    data Olimpiade Dunia. Kami menyajikan berbagai analisis dan visualisasi interaktif tentang
    pelaksanaan Olimpiade.
    
    Fitur-fitur utama website:
    * 📊 Visualisasi data interaktif
    * 📈 Analisis tren temporal
    * 🔍 Eksplorasi korelasi antar parameter
    * 📱 Tampilan responsif dan user-friendly
    """)
    
    st.markdown("---")

    # Informasi Dataset
    st.header("📂 Tentang Dataset")
    st.write("""
    Ini adalah kumpulan data historis tentang Olimpiade, termasuk semua Olimpiade dari Athena 1896 hingga Beijing 2022. Kumpulan data tersebut mencakup hasil, medali, atlet, dan tuan rumah. Data tersebut dibuat dari Olimpiade .
    Lebih dari 21.000 medali, 162.000 hasil, 74.000 atlet, 20.000 biografi, dan 53 tuan rumah Olimpiade Musim Panas dan Musim Dingin dapat ditemukan di sini..
    """)

    # Menampilkan preview dataset
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("Struktur Dataset:")
        st.markdown("""
        Dataset ini mencakup parameter berikut:
        * olympic_athletes.csv: informasi pribadi tentang atlet (nama, biografi singkat, dll.)
        * olympic_medals.csv: informasi umum tentang peraih medali (Atlet atau Tim)
        * olympic_hosts.csv: tuan rumah (tahun, kota, negara, dll)
        * olympic_results.csv: informasi umum tentang hasil (Atlet atau Tim)
        """)
    
    with col4:
        st.subheader("Sumber Dataset:")
        st.markdown("""
        Dataset ini diperoleh dari Kaggle, Olympic Summer & Winter Games, 1896-2022.
        
        [Link Dataset](https://www.kaggle.com/datasets/piterfm/olympic-games-medals-19862018/data)
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>© 2024 Group 6 Data Aquisition-B . All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Dashboard Olympic Games 1896-2022",
        page_icon="🏅",
        layout="wide"
    )
    
    # Navigation
    pages = {
        "Home": show_home,
        "Input Dataset": load_dataset,
        "Preprocess Data": detailed_eda,
        "Analisis Data": perform_analysis,
        "Visualisasi Data": create_visualization
    }
    
    page = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Initialize session state for data persistence
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Show home page or process other pages
    if page == "Home":
        show_home()
    else:
        # Load data if not already loaded
        if page == "Input Dataset":
            st.session_state.data = load_dataset()
        
        # Process data if available
        if st.session_state.data is not None:
            if page == "Preprocess Data":
                st.session_state.processed_data = detailed_eda(st.session_state.data)
            elif page == "Analisis Data":
                perform_analysis(st.session_state.processed_data if 'processed_data' in st.session_state else st.session_state.data)
            elif page == "Visualisasi Data":
                create_visualization(st.session_state.processed_data if 'processed_data' in st.session_state else st.session_state.data)
        else:
            if page != "Input Dataset":
                st.warning("Please upload datasets first.")

if __name__ == "__main__":
    main()