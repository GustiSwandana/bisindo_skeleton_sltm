from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / 'backend' / 'artifacts'
REPORTS_DIR = ROOT / 'reports'
METADATA_PATH = ARTIFACTS_DIR / 'label_map.json'
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / 'confusion_matrix.json'
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / 'classification_report.json'
OUTPUT_DOCX = REPORTS_DIR / 'Perhitungan_Manual_BISINDO.docx'
OUTPUT_MD = REPORTS_DIR / 'Perhitungan_Manual_BISINDO.md'


def pct(value: float) -> str:
    return f'{value * 100:.2f}%'


def build_classification_table_lines(classification_report_payload: dict | None) -> list[str]:
    if not classification_report_payload:
        return []

    lines = [
        '',
        '## 8. Classification Report per Kelas',
        '| Kelas | Precision | Recall | F1-score | Support |',
        '|---|---:|---:|---:|---:|',
    ]

    for row in classification_report_payload.get('per_class', []):
        lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['support']} |"
        )

    macro_avg = classification_report_payload.get('macro_avg', {})
    weighted_avg = classification_report_payload.get('weighted_avg', {})
    accuracy = classification_report_payload.get('accuracy', 0.0)
    total_support = macro_avg.get('support', 0)

    lines.extend(
        [
            f"| accuracy | - | - | {accuracy:.4f} | {total_support} |",
            f"| macro avg | {macro_avg.get('precision', 0.0):.4f} | {macro_avg.get('recall', 0.0):.4f} | {macro_avg.get('f1_score', 0.0):.4f} | {macro_avg.get('support', 0)} |",
            f"| weighted avg | {weighted_avg.get('precision', 0.0):.4f} | {weighted_avg.get('recall', 0.0):.4f} | {weighted_avg.get('f1_score', 0.0):.4f} | {weighted_avg.get('support', 0)} |",
        ]
    )
    return lines


def build_thesis_explanation_lines(metadata: dict, classification_report_payload: dict | None = None) -> list[str]:
    sequence_length = int(metadata['sequence_length'])
    feature_dim = int(metadata['feature_dim'])
    class_count = len(metadata['class_names'])
    accuracy = float(metadata['final_val_accuracy'])
    macro_f1 = 0.0
    weighted_f1 = 0.0
    if classification_report_payload:
        macro_f1 = float(classification_report_payload.get('macro_avg', {}).get('f1_score', 0.0))
        weighted_f1 = float(classification_report_payload.get('weighted_avg', {}).get('f1_score', 0.0))

    return [
        '',
        '## 2. Uraian Metode dalam Bahasa Laporan',
        'Penelitian ini mengimplementasikan sistem pengenalan alfabet Bahasa Isyarat BISINDO berbasis citra statis dengan pendekatan hibrida skeleton dan Long Short-Term Memory (LSTM). Tahap awal sistem adalah ekstraksi ciri tangan menggunakan MediaPipe Hands untuk memperoleh titik-titik landmark anatomi tangan. Setiap tangan direpresentasikan oleh 21 landmark tiga dimensi, sehingga ketika sistem mendeteksi dua tangan, total landmark yang digunakan menjadi 42 titik.',
        'Landmark yang diperoleh tidak langsung digunakan sebagai masukan model. Tahap praproses dilakukan melalui translasi terhadap titik wrist, rotasi orientasi terhadap sumbu acuan wrist ke middle metacarpophalangeal joint, dan normalisasi skala berdasarkan geometri telapak tangan. Tahap ini bertujuan untuk mengurangi variasi yang disebabkan oleh perbedaan posisi tangan, kemiringan, serta jarak tangan terhadap kamera.',
        f'Setelah normalisasi, setiap landmark diubah menjadi vektor fitur berdimensi {feature_dim}, yang terdiri atas koordinat ternormalisasi, vektor relatif terhadap parent joint, jarak terhadap wrist, serta jarak terhadap centroid. Dengan demikian, satu sampel citra direpresentasikan sebagai matriks berukuran {sequence_length} x {feature_dim}. Representasi ini selanjutnya diperlakukan sebagai pseudo-sequence agar hubungan antarlandmark dapat dipelajari secara berurutan oleh model LSTM meskipun data yang digunakan berupa gambar statis, bukan video.',
        'Pemilihan arsitektur LSTM pada penelitian ini didasarkan pada kemampuannya dalam mempelajari dependensi antar elemen urutan. Pada implementasinya, model menggunakan Bidirectional LSTM sehingga proses pembelajaran dilakukan dari arah awal ke akhir dan dari arah akhir ke awal sequence. Pendekatan ini memungkinkan model menangkap keterkaitan spasial antarlandmark dengan konteks yang lebih lengkap dibandingkan LSTM satu arah.',
        f'Keluaran dari lapisan Bidirectional LSTM kemudian diteruskan ke lapisan dense untuk melakukan pemetaan fitur menuju {class_count} kelas alfabet BISINDO. Lapisan output menggunakan fungsi aktivasi softmax untuk menghasilkan probabilitas setiap kelas. Kelas dengan probabilitas tertinggi ditetapkan sebagai hasil prediksi akhir sistem.',
        f'Berdasarkan hasil pelatihan terakhir, model memperoleh akurasi validasi sebesar {pct(accuracy)}. Selain itu, evaluasi multi-kelas menunjukkan nilai macro F1-score sebesar {macro_f1:.4f} dan weighted F1-score sebesar {weighted_f1:.4f}. Nilai tersebut menunjukkan bahwa model tidak hanya memiliki ketepatan prediksi yang tinggi secara umum, tetapi juga cukup seimbang dalam mengenali berbagai kelas alfabet BISINDO.',
        'Secara keseluruhan, alur kerja sistem terdiri atas empat tahap utama, yaitu akuisisi citra, ekstraksi skeleton tangan, pembentukan pseudo-sequence fitur, dan klasifikasi dengan model Bidirectional LSTM. Rangkaian proses ini dirancang agar sistem mampu mengenali alfabet BISINDO secara otomatis baik dari citra unggahan maupun frame webcam secara real-time.',
    ]


def build_report_lines(
    metadata: dict,
    confusion_matrix_payload: dict | None = None,
    classification_report_payload: dict | None = None,
) -> list[str]:
    class_count = len(metadata['class_names'])
    train_samples = int(metadata['train_samples'])
    val_samples = int(metadata['val_samples'])
    train_skipped = int(metadata['train_skipped'])
    val_skipped = int(metadata['val_skipped'])
    train_total = train_samples + train_skipped
    val_total = val_samples + val_skipped
    train_skip_rate = train_skipped / train_total if train_total else 0.0
    val_skip_rate = val_skipped / val_total if val_total else 0.0
    sequence_length = int(metadata['sequence_length'])
    feature_dim = int(metadata['feature_dim'])
    total_input_features = sequence_length * feature_dim

    lstm1_units = 128
    lstm2_units = 64
    dense_units = 64

    lstm1_single = 4 * ((feature_dim + lstm1_units) * lstm1_units + lstm1_units)
    lstm1_bi = 2 * lstm1_single
    lstm2_input_dim = lstm1_units * 2
    lstm2_single = 4 * ((lstm2_input_dim + lstm2_units) * lstm2_units + lstm2_units)
    lstm2_bi = 2 * lstm2_single
    dense1_params = (lstm2_units * 2 * dense_units) + dense_units
    output_params = (dense_units * class_count) + class_count
    total_params = lstm1_bi + lstm2_bi + dense1_params + output_params

    confusion_summary_lines: list[str] = []
    if confusion_matrix_payload:
        matrix = confusion_matrix_payload.get('matrix', [])
        diagonal = sum(
            matrix[index][index]
            for index in range(min(len(matrix), len(matrix[0]) if matrix else 0))
        )
        total_samples_cm = int(confusion_matrix_payload.get('total_samples', 0))
        confusion_summary_lines = [
            '',
            '## 7. Confusion Matrix',
            f'- Total sampel validasi pada confusion matrix: {total_samples_cm}',
            f'- Prediksi benar pada diagonal utama: {diagonal}',
            f'- Gambar confusion matrix: {confusion_matrix_payload.get("image_path", "")}',
            '- Baris menunjukkan label sebenarnya, kolom menunjukkan label hasil prediksi.',
            '- Nilai diagonal yang tinggi menandakan model mampu mengklasifikasikan kelas tersebut dengan benar.',
        ]

    classification_table_lines = build_classification_table_lines(classification_report_payload)
    thesis_explanation_lines = build_thesis_explanation_lines(
        metadata,
        classification_report_payload=classification_report_payload,
    )

    wrist = (0.50, 0.60, 0.00)
    index_mcp = (0.45, 0.40, -0.01)
    middle_mcp = (0.55, 0.35, -0.02)
    pinky_mcp = (0.70, 0.43, -0.02)

    translated_index = (
        index_mcp[0] - wrist[0],
        index_mcp[1] - wrist[1],
        index_mcp[2] - wrist[2],
    )
    translated_middle = (
        middle_mcp[0] - wrist[0],
        middle_mcp[1] - wrist[1],
        middle_mcp[2] - wrist[2],
    )
    translated_pinky = (
        pinky_mcp[0] - wrist[0],
        pinky_mcp[1] - wrist[1],
        pinky_mcp[2] - wrist[2],
    )

    angle = math.atan2(translated_middle[1], translated_middle[0])
    target_angle = -math.pi / 2
    rotation = target_angle - angle
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)

    def rotate_xy(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        return (
            x * cos_theta - y * sin_theta,
            x * sin_theta + y * cos_theta,
            z,
        )

    rotated_index = rotate_xy(translated_index)
    rotated_middle = rotate_xy(translated_middle)
    rotated_pinky = rotate_xy(translated_pinky)

    palm_height = math.sqrt(sum(value * value for value in rotated_middle))
    palm_width = math.sqrt(
        (rotated_index[0] - rotated_pinky[0]) ** 2
        + (rotated_index[1] - rotated_pinky[1]) ** 2
        + (rotated_index[2] - rotated_pinky[2]) ** 2
    )
    max_radius = max(
        math.sqrt(sum(value * value for value in rotated_index)),
        math.sqrt(sum(value * value for value in rotated_middle)),
        math.sqrt(sum(value * value for value in rotated_pinky)),
    )
    scale = max(palm_height, palm_width, max_radius, 1e-6)

    normalized_index = tuple(value / scale for value in rotated_index)
    normalized_middle = tuple(value / scale for value in rotated_middle)
    normalized_pinky = tuple(value / scale for value in rotated_pinky)

    centroid = (
        (normalized_index[0] + normalized_middle[0] + normalized_pinky[0]) / 3,
        (normalized_index[1] + normalized_middle[1] + normalized_pinky[1]) / 3,
        (normalized_index[2] + normalized_middle[2] + normalized_pinky[2]) / 3,
    )

    wrist_distance_index = math.sqrt(sum(value * value for value in normalized_index))
    centroid_distance_index = math.sqrt(
        (normalized_index[0] - centroid[0]) ** 2
        + (normalized_index[1] - centroid[1]) ** 2
        + (normalized_index[2] - centroid[2]) ** 2
    )

    lines = [
        '# Perhitungan Manual Aplikasi Pendeteksian Alfabet BISINDO',
        '',
        f'Tanggal ekspor: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '## 1. Ringkasan Model dan Dataset',
        f'- Folder train: {metadata["train_dir"]}',
        f'- Folder validasi: {metadata["val_dir"]}',
        f'- Jumlah kelas alfabet: {class_count}',
        f'- Sequence length model: {sequence_length}',
        f'- Feature dimension per timestep: {feature_dim}',
        f'- Total fitur per sampel: {sequence_length} x {feature_dim} = {total_input_features}',
        f'- Sampel train terpakai: {train_samples}',
        f'- Sampel train di-skip: {train_skipped}',
        f'- Total train awal: {train_total}',
        f'- Persentase skip train: {pct(train_skip_rate)}',
        f'- Sampel validasi terpakai: {val_samples}',
        f'- Sampel validasi di-skip: {val_skipped}',
        f'- Total validasi awal: {val_total}',
        f'- Persentase skip validasi: {pct(val_skip_rate)}',
        f'- Akurasi validasi akhir: {pct(float(metadata["final_val_accuracy"]))}',
        f'- Loss validasi akhir: {float(metadata["final_val_loss"]):.6f}',
        '',
        *thesis_explanation_lines,
        '',
        '## 3. Perhitungan Manual Preprocessing Skeleton',
        'Pada aplikasi ini, setiap gambar dapat dibaca sampai 2 tangan. Tiap tangan memiliki 21 landmark MediaPipe, sehingga sequence length = 2 x 21 = 42 timestep.',
        '',
        '### 3.1 Translasi terhadap wrist',
        'Rumus: translated = landmark - wrist',
        f'- Wrist = {wrist}',
        f'- Index MCP = {index_mcp}',
        f'- Hasil translasi Index MCP = {translated_index}',
        f'- Middle MCP = {middle_mcp}',
        f'- Hasil translasi Middle MCP = {translated_middle}',
        '',
        '### 3.2 Rotasi orientasi tangan',
        'Aplikasi memutar sumbu x-y agar arah wrist ke middle MCP menjadi tegak. Ini membuat model lebih stabil terhadap tangan yang miring.',
        f'- Sudut awal = atan2({translated_middle[1]:.4f}, {translated_middle[0]:.4f}) = {angle:.6f} rad',
        f'- Target sudut = -pi/2 = {-math.pi / 2:.6f} rad',
        f'- Rotasi yang dipakai = target - sudut awal = {rotation:.6f} rad',
        f'- cos(theta) = {cos_theta:.6f}',
        f'- sin(theta) = {sin_theta:.6f}',
        f'- Index MCP setelah rotasi = ({rotated_index[0]:.6f}, {rotated_index[1]:.6f}, {rotated_index[2]:.6f})',
        f'- Middle MCP setelah rotasi = ({rotated_middle[0]:.6f}, {rotated_middle[1]:.6f}, {rotated_middle[2]:.6f})',
        '',
        '### 3.3 Penentuan skala normalisasi',
        'Aplikasi memilih skala terbesar dari tinggi telapak, lebar telapak, dan radius maksimum.',
        f'- Palm height = ||middle_mcp|| = {palm_height:.6f}',
        f'- Palm width = ||index_mcp - pinky_mcp|| = {palm_width:.6f}',
        f'- Max radius = {max_radius:.6f}',
        f'- Skala akhir = max(palm_height, palm_width, max_radius) = {scale:.6f}',
        '',
        '### 3.4 Normalisasi satu landmark contoh',
        'Rumus: normalized = translated_rotated / scale',
        f'- Normalized Index MCP = ({normalized_index[0]:.6f}, {normalized_index[1]:.6f}, {normalized_index[2]:.6f})',
        f'- Normalized Middle MCP = ({normalized_middle[0]:.6f}, {normalized_middle[1]:.6f}, {normalized_middle[2]:.6f})',
        '',
        '### 3.5 Penyusunan feature vector per landmark',
        'Untuk model terbaru, satu landmark menghasilkan 8 fitur:',
        '- normalized_x',
        '- normalized_y',
        '- normalized_z',
        '- parent_vector_x',
        '- parent_vector_y',
        '- parent_vector_z',
        '- distance_to_wrist',
        '- distance_to_centroid',
        '',
        'Contoh perhitungan untuk landmark Index MCP (parent = wrist):',
        f'- Parent vector = normalized_index - wrist_normalized = ({normalized_index[0]:.6f}, {normalized_index[1]:.6f}, {normalized_index[2]:.6f})',
        f'- Centroid contoh (3 titik ilustrasi) = ({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})',
        f'- Distance to wrist = {wrist_distance_index:.6f}',
        f'- Distance to centroid = {centroid_distance_index:.6f}',
        '',
        'Maka feature vector ilustratif untuk Index MCP adalah:',
        f'[{normalized_index[0]:.6f}, {normalized_index[1]:.6f}, {normalized_index[2]:.6f}, {normalized_index[0]:.6f}, {normalized_index[1]:.6f}, {normalized_index[2]:.6f}, {wrist_distance_index:.6f}, {centroid_distance_index:.6f}]',
        '',
        '## 4. Perhitungan Ukuran Input Model',
        f'- Maksimum tangan terbaca = 2',
        f'- Landmark per tangan = 21',
        f'- Sequence length = 2 x 21 = {sequence_length}',
        f'- Feature dim = {feature_dim}',
        f'- Total nilai input per sampel = {sequence_length} x {feature_dim} = {total_input_features}',
        '',
        '## 5. Perhitungan Manual Parameter LSTM',
        'Rumus parameter satu layer LSTM: 4 x ((input_dim + units) x units + units)',
        '',
        '### 5.1 Bidirectional LSTM pertama',
        f'- input_dim = {feature_dim}',
        f'- units = {lstm1_units}',
        f'- Parameter satu arah = 4 x (({feature_dim} + {lstm1_units}) x {lstm1_units} + {lstm1_units}) = {lstm1_single}',
        f'- Karena bidirectional, total = 2 x {lstm1_single} = {lstm1_bi}',
        '',
        '### 5.2 Bidirectional LSTM kedua',
        f'- input_dim = 2 x {lstm1_units} = {lstm2_input_dim}',
        f'- units = {lstm2_units}',
        f'- Parameter satu arah = 4 x (({lstm2_input_dim} + {lstm2_units}) x {lstm2_units} + {lstm2_units}) = {lstm2_single}',
        f'- Karena bidirectional, total = 2 x {lstm2_single} = {lstm2_bi}',
        '',
        '### 5.3 Dense layer',
        f'- Dense(64): ({lstm2_units * 2} x {dense_units}) + {dense_units} = {dense1_params}',
        f'- Dense output(26): ({dense_units} x {class_count}) + {class_count} = {output_params}',
        '',
        '### 5.4 Total parameter model',
        f'- Total = {lstm1_bi} + {lstm2_bi} + {dense1_params} + {output_params} = {total_params}',
        '',
        '## 6. Perhitungan Prediksi Softmax',
        'Output model adalah 26 nilai logit yang diubah menjadi probabilitas dengan softmax:',
        'softmax(z_i) = exp(z_i) / sum(exp(z_j))',
        '',
        'Contoh sederhana jika tiga logit terbesar adalah [2.80, 1.20, 0.50]:',
        '- exp(2.80) = 16.4446',
        '- exp(1.20) = 3.3201',
        '- exp(0.50) = 1.6487',
        '- Total = 21.4134',
        '- Probabilitas kelas-1 = 16.4446 / 21.4134 = 0.7680 = 76.80%',
        *confusion_summary_lines,
        *classification_table_lines,
        '',
        '## 9. Kesimpulan untuk Laporan',
        f'Aplikasi membaca maksimum 2 tangan, menyusun sequence sepanjang {sequence_length} timestep, dan menghasilkan {feature_dim} fitur per timestep sehingga total input per sampel adalah {total_input_features} nilai.',
        f'Dari metadata training terbaru, model mencapai akurasi validasi {pct(float(metadata["final_val_accuracy"]))} dengan total parameter jaringan sekitar {total_params:,} parameter trainable.',
        'Perhitungan manual ini dapat langsung dimasukkan ke bab metode, implementasi, dan analisis hasil pada laporan skripsi atau proyek.',
    ]
    return lines


def build_document_xml(lines: list[str]) -> str:
    paragraphs = []
    for line in lines:
        if line == '':
            paragraphs.append('<w:p/>')
            continue
        text = escape(line)
        paragraphs.append(
            '<w:p><w:r><w:t xml:space="preserve">' + text + '</w:t></w:r></w:p>'
        )

    body = ''.join(paragraphs) + (
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        '</w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        '<w:body>' + body + '</w:body></w:document>'
    )


def write_docx(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_xml = build_document_xml(lines)

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Perhitungan Manual BISINDO</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}</dcterms:modified>
</cp:coreProperties>'''

    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>'''

    with ZipFile(output_path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', content_types)
        archive.writestr('_rels/.rels', rels)
        archive.writestr('docProps/core.xml', core)
        archive.writestr('docProps/app.xml', app)
        archive.writestr('word/document.xml', document_xml)


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    confusion_matrix_payload = None
    classification_report_payload = None
    if CONFUSION_MATRIX_PATH.exists():
        confusion_matrix_payload = json.loads(CONFUSION_MATRIX_PATH.read_text(encoding='utf-8'))
    if CLASSIFICATION_REPORT_PATH.exists():
        classification_report_payload = json.loads(CLASSIFICATION_REPORT_PATH.read_text(encoding='utf-8'))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = build_report_lines(
        metadata,
        confusion_matrix_payload=confusion_matrix_payload,
        classification_report_payload=classification_report_payload,
    )
    OUTPUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    output_docx = OUTPUT_DOCX
    try:
        write_docx(lines, output_docx)
    except PermissionError:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_docx = REPORTS_DIR / f'Perhitungan_Manual_BISINDO_{timestamp}.docx'
        write_docx(lines, output_docx)
    print(f'Report markdown: {OUTPUT_MD}')
    print(f'Report docx: {output_docx}')


if __name__ == '__main__':
    main()
