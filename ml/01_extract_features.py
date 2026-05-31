
import argparse
import os

import numpy as np
import pandas as pd
from scapy.all import DNS, ICMP, IP, TCP, UDP, rdpcap

# Ham pcap dosyalarının ve çıktı dizinlerinin yolları
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "features.csv")
EXTRACTION_SUMMARY_CSV = os.path.join(PROCESSED_DIR, "extraction_summary.csv")

# Varsayılan pencere boyutu — 1 saniyelik dilimler halinde analiz yapıyoruz
DEFAULT_WINDOW_SIZE = 1.0


def safe_div(a, b):
    # Sıfıra bölme hatasını önlemek için bu yardımcı fonksiyonu kullanıyoruz
    return a / b if b != 0 else 0


def entropy(values):
    # Bir dizi değerin bilgi entropisi — port çeşitliliği gibi metriklerde işimize yarıyor
    if len(values) == 0:
        return 0
    counts = pd.Series(values).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum())


def get_tcp_flags(pkt):
    # TCP bayraklarını bit maskeleriyle tek tek okuyoruz
    flags = {
        "syn": 0,
        "ack": 0,
        "fin": 0,
        "rst": 0,
        "psh": 0,
        "urg": 0,
    }

    if TCP in pkt:
        tcp_flags = int(pkt[TCP].flags)
        flags["fin"] = 1 if tcp_flags & 0x01 else 0
        flags["syn"] = 1 if tcp_flags & 0x02 else 0
        flags["rst"] = 1 if tcp_flags & 0x04 else 0
        flags["psh"] = 1 if tcp_flags & 0x08 else 0
        flags["ack"] = 1 if tcp_flags & 0x10 else 0
        flags["urg"] = 1 if tcp_flags & 0x20 else 0

    return flags


def extract_features_from_pcap(
    pcap_path,
    label,
    window_size=DEFAULT_WINDOW_SIZE,
    return_summary=False,
):
    # Dosya özet bilgisini baştan hazırlıyoruz, hata olursa buraya yazacağız
    summary = {
        "file_name": os.path.basename(pcap_path),
        "label": label,
        "file_size_bytes": os.path.getsize(pcap_path) if os.path.exists(pcap_path) else 0,
        "total_packets": 0,
        "ip_packets": 0,
        "feature_rows_before_dropna": 0,
        "status": "ok",
        "error": "",
    }

    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        # Dosya okunamazsa hata kaydedip boş DataFrame dönüyoruz
        summary["status"] = "read_error"
        summary["error"] = str(e)
        print(f"[HATA] Dosya okunamadi: {pcap_path} -> {e}")
        empty_df = pd.DataFrame()
        return (empty_df, summary) if return_summary else empty_df

    summary["total_packets"] = len(packets)
    packet_rows = []

    # Her paketi dolaşıp IP katmanı olanları işliyoruz
    for pkt in packets:
        if IP not in pkt:
            continue  # IP paketi değilse atla

        time_value = float(pkt.time)
        length = len(pkt)

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst

        src_port = 0
        dst_port = 0

        is_tcp = 0
        is_udp = 0
        is_icmp = 0
        is_dns = 0

        # Protokolü kontrol edip port bilgilerini alıyoruz
        if TCP in pkt:
            is_tcp = 1
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
        elif UDP in pkt:
            is_udp = 1
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
        elif ICMP in pkt:
            is_icmp = 1

        if DNS in pkt:
            is_dns = 1

        flags = get_tcp_flags(pkt)

        packet_rows.append(
            {
                "time": time_value,
                "length": length,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "is_tcp": is_tcp,
                "is_udp": is_udp,
                "is_icmp": is_icmp,
                "is_dns": is_dns,
                "syn": flags["syn"],
                "ack": flags["ack"],
                "fin": flags["fin"],
                "rst": flags["rst"],
                "psh": flags["psh"],
                "urg": flags["urg"],
            }
        )

    summary["ip_packets"] = len(packet_rows)

    if len(packet_rows) == 0:
        summary["status"] = "no_ip_packets"
        print(f"[UYARI] IP paketi bulunamadi: {pcap_path}")
        empty_df = pd.DataFrame()
        return (empty_df, summary) if return_summary else empty_df

    df = pd.DataFrame(packet_rows)
    start_time = df["time"].min()
    # Her paketi hangi zaman dilimine (pencereye) ait olduğunu hesaplıyoruz
    df["window_id"] = ((df["time"] - start_time) // window_size).astype(int)

    feature_rows = []

    # Her zaman penceresi için istatistiksel özellikler çıkarıyoruz
    for window_id, group in df.groupby("window_id"):
        packet_count = len(group)

        duration = group["time"].max() - group["time"].min()
        if duration <= 0:
            duration = window_size  # Tek paket varsa süreyi pencere boyutuna eşitliyoruz

        # Paketler arası geliş sürelerini hesaplıyoruz
        interarrival = group["time"].sort_values().diff().dropna()

        tcp_count = int(group["is_tcp"].sum())
        udp_count = int(group["is_udp"].sum())
        icmp_count = int(group["is_icmp"].sum())
        dns_count = int(group["is_dns"].sum())

        syn_count = int(group["syn"].sum())
        ack_count = int(group["ack"].sum())
        fin_count = int(group["fin"].sum())
        rst_count = int(group["rst"].sum())
        psh_count = int(group["psh"].sum())
        urg_count = int(group["urg"].sum())

        # Bu penceredeki tüm özellikleri bir satır olarak ekliyoruz
        feature_rows.append(
            {
                "file_name": os.path.basename(pcap_path),
                "window_id": int(window_id),
                "packet_count": packet_count,
                "byte_count": int(group["length"].sum()),
                "avg_packet_len": float(group["length"].mean()),
                "std_packet_len": float(group["length"].std()) if packet_count > 1 else 0,
                "min_packet_len": int(group["length"].min()),
                "max_packet_len": int(group["length"].max()),
                "tcp_count": tcp_count,
                "udp_count": udp_count,
                "icmp_count": icmp_count,
                "dns_count": dns_count,
                "syn_count": syn_count,
                "ack_count": ack_count,
                "fin_count": fin_count,
                "rst_count": rst_count,
                "psh_count": psh_count,
                "urg_count": urg_count,
                "unique_src_ip": int(group["src_ip"].nunique()),
                "unique_dst_ip": int(group["dst_ip"].nunique()),
                "unique_src_port": int(group["src_port"].nunique()),
                "unique_dst_port": int(group["dst_port"].nunique()),
                "src_ip_entropy": entropy(group["src_ip"]),
                "dst_ip_entropy": entropy(group["dst_ip"]),
                "src_port_entropy": entropy(group["src_port"]),
                "dst_port_entropy": entropy(group["dst_port"]),
                "packets_per_second": safe_div(packet_count, duration),
                "bytes_per_second": safe_div(group["length"].sum(), duration),
                "avg_interarrival": float(interarrival.mean()) if len(interarrival) > 0 else 0,
                "std_interarrival": float(interarrival.std()) if len(interarrival) > 1 else 0,
                "tcp_ratio": safe_div(tcp_count, packet_count),
                "udp_ratio": safe_div(udp_count, packet_count),
                "icmp_ratio": safe_div(icmp_count, packet_count),
                "dns_ratio": safe_div(dns_count, packet_count),
                "syn_ratio": safe_div(syn_count, packet_count),
                "ack_ratio": safe_div(ack_count, packet_count),
                "fin_ratio": safe_div(fin_count, packet_count),
                "rst_ratio": safe_div(rst_count, packet_count),
                "psh_ratio": safe_div(psh_count, packet_count),
                "urg_ratio": safe_div(urg_count, packet_count),
                "label": label,
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    summary["feature_rows_before_dropna"] = len(feature_df)

    return (feature_df, summary) if return_summary else feature_df


def iter_pcap_files(folder_path):
    # Klasördeki tüm .pcap ve .pcapng dosyalarını sıralı şekilde buluyoruz
    for root, _, files in os.walk(folder_path):
        for file_name in sorted(files):
            if file_name.lower().endswith((".pcap", ".pcapng")):
                yield os.path.join(root, file_name)


def process_folder(folder_path, label, window_size):
    # Bir klasördeki tüm pcap dosyalarını işleyip özellik tablolarını birleştiriyoruz
    all_dataframes = []
    summaries = []

    if not os.path.exists(folder_path):
        print(f"[UYARI] Klasor yok: {folder_path}")
        return all_dataframes, summaries

    for file_path in iter_pcap_files(folder_path):
        print(f"[OKUNUYOR] {file_path}")
        df, summary = extract_features_from_pcap(
            file_path,
            label=label,
            window_size=window_size,
            return_summary=True,
        )

        if not df.empty:
            all_dataframes.append(df)

        summaries.append(summary)

    return all_dataframes, summaries


def parse_args():
    parser = argparse.ArgumentParser(description="PCAP/PCAPNG dosyalarindan pencere bazli ozellik cikarir.")
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE, help="Saniye cinsinden pencere boyutu.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Normal ve anomali klasörlerini ayrı ayrı işliyoruz, etiket de buna göre veriliyor
    normal_dir = os.path.join(RAW_DIR, "normal")
    anomaly_dir = os.path.join(RAW_DIR, "anomaly")

    normal_dfs, normal_summaries = process_folder(normal_dir, label=0, window_size=args.window_size)
    anomaly_dfs, anomaly_summaries = process_folder(anomaly_dir, label=1, window_size=args.window_size)

    all_dfs = normal_dfs + anomaly_dfs
    all_summaries = normal_summaries + anomaly_summaries

    if len(all_dfs) == 0:
        print("[HATA] Hic ozellik cikarilamadi.")
        if all_summaries:
            pd.DataFrame(all_summaries).to_csv(EXTRACTION_SUMMARY_CSV, index=False)
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    rows_before_dropna = len(final_df)

    # Sonsuz ve NaN değerleri temizliyoruz
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.dropna()

    final_df.to_csv(OUTPUT_CSV, index=False)

    # Her dosya için özet raporu hazırlıyoruz
    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        final_rows = final_df.groupby("file_name").size().reset_index(name="feature_rows_after_dropna")
        summary_df = summary_df.merge(final_rows, on="file_name", how="left")
        summary_df["feature_rows_after_dropna"] = summary_df["feature_rows_after_dropna"].fillna(0).astype(int)
        summary_df["window_size"] = args.window_size
        summary_df.to_csv(EXTRACTION_SUMMARY_CSV, index=False)

    print("\n[TAMAM] Oznitelikler cikarildi.")
    print(f"[PENCERE BOYUTU] {args.window_size} saniye")
    print(f"[KAYIT] {OUTPUT_CSV}")
    print(f"[OZET] {EXTRACTION_SUMMARY_CSV}")
    print(f"[DROPNA ONCESI SATIR] {rows_before_dropna}")
    print(f"[SATIR SAYISI] {len(final_df)}")
    print(f"[NORMAL SATIR] {(final_df['label'] == 0).sum()}")
    print(f"[ANOMALI SATIR] {(final_df['label'] == 1).sum()}")


if __name__ == "__main__":
    main()
