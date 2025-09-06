# Энэхүү файл нь kmeans алгоритмын гаралтын лог файлыг уншиж, шаардлагатай мэдээллийг цуглуулан Excel файлд хадгалах зориулалттай.
# kmeans_keq5output.txt файлыг уншиж, MSAVI, кластер төвүүд, Moran's I, p-value, ургамлын кластерын мэдээллийг цуглуулна.

# import re
# import pandas as pd

# def parse_log_file(filepath):
#     with open(filepath, 'r') as f:
#         content = f.read()

#     # Split by each block
#     blocks = content.strip().split("Processing tifs corresponding to:")

#     records = []

#     for block in blocks[1:]:  # skip first if empty
#         lines = block.strip().splitlines()
#         try:
#             msavi_file = lines[0].strip()
#             image_name = lines[1].strip()
#             label = lines[2].strip()
#             msavi_path = lines[3].replace("Loaded MSAVI from: ", "").strip()
#             msavi_mean = float(lines[4].strip())

#             # Cluster centers
#             cluster_centers_line = lines[5]
#             centers = re.findall(r"[-]?\d+\.\d+", cluster_centers_line)
#             centers = list(map(float, centers))

#             # Cluster coverages
#             coverage_lines = lines[6:9]
#             coverages = []
#             for line in coverage_lines:
#                 match = re.search(r"coverage: ([\d.]+) %", line)
#                 coverages.append(float(match.group(1)) if match else None)

#             # Moran's I
#             moran_line = lines[9]
#             moran_i, p_value = map(float, re.findall(r"[-]?\d+\.\d+", moran_line))

#             # Weed cluster
#             weed_value = float(re.search(r"value: ([\d.]+)", lines[10]).group(1))
#             weed_coverage = float(re.search(r"([\d.]+)%", lines[11]).group(1))

#             # Plot path
#             plot_path = lines[12].replace("Plot saved to: ", "").strip()

#             records.append({
#                 "Image Name": image_name,
#                 "Label": label,
#                 "MSAVI Path": msavi_path,
#                 "Mean MSAVI": msavi_mean,
#                 "Cluster Center 0": centers[0],
#                 "Cluster Center 1": centers[1],
#                 "Cluster Center 2": centers[2],
#                 "Coverage 0 (%)": coverages[0],
#                 "Coverage 1 (%)": coverages[1],
#                 "Coverage 2 (%)": coverages[2],
#                 "Moran's I": moran_i,
#                 "p-value": p_value,
#                 "Weed Cluster Value": weed_value,
#                 "Weed Coverage (%)": weed_coverage,
#                 "Plot Path": plot_path
#             })
#         except Exception as e:
#             print(f"Error parsing block:\n{block}\nError: {e}")

#     return pd.DataFrame(records)

# if __name__ == "__main__":
#     df = parse_log_file("kmeans_output.txt")  # Change this to your file path
#     df.to_excel("weed_analysis_summary.xlsx", index=False)
#     print("Saved data to weed_analysis_summary.xlsx")

# import re
# import pandas as pd

# def parse_log_file(filepath):
#     with open(filepath, 'r') as f:
#         content = f.read()

#     blocks = content.strip().split("Processing tifs corresponding to:")

#     records = []

#     for block in blocks[1:]:
#         lines = block.strip().splitlines()
#         try:
#             msavi_file = lines[0].strip()
#             image_name = lines[1].strip()
#             label = lines[2].strip()
#             msavi_path = lines[3].replace("Loaded MSAVI from: ", "").strip()
#             msavi_mean = float(lines[4].strip())

#             # Extract cluster centers
#             center_line = lines[5]
#             centers = re.findall(r"[-]?\d+\.\d+", center_line)
#             centers = list(map(float, centers))

#             # Extract coverages from lines 6, 7, 8 and match them with center order
#             cluster_info = []
#             for line in lines[6:9]:
#                 center_match = re.search(r"center=([-]?\d+\.\d+)", line)
#                 coverage_match = re.search(r"coverage: ([\d.]+) %", line)
#                 if center_match and coverage_match:
#                     center = float(center_match.group(1))
#                     coverage = float(coverage_match.group(1))
#                     cluster_info.append((center, coverage))

#             # Sort cluster info by center intensity
#             cluster_info.sort(key=lambda x: x[0])

#             # Moran's I and p-value
#             moran_i, p_value = map(float, re.findall(r"[-]?\d+\.\d+", lines[9]))

#             # Weed cluster info
#             weed_value = float(re.search(r"value: ([\d.]+)", lines[10]).group(1))
#             weed_coverage = float(re.search(r"([\d.]+)%", lines[11]).group(1))

#             # Plot path
#             plot_path = lines[12].replace("Plot saved to: ", "").strip()

#             # Construct row
#             record = {
#                 "Image Name": image_name,
#                 "Label": label,
#                 "MSAVI Path": msavi_path,
#                 "Mean MSAVI": msavi_mean,
#                 "Moran's I": moran_i,
#                 "p-value": p_value,
#                 "Weed Cluster Value": weed_value,
#                 "Weed Coverage (%)": weed_coverage,
#                 "Plot Path": plot_path
#             }

#             # Add sorted cluster info
#             for i, (center, coverage) in enumerate(cluster_info):
#                 record[f"Cluster Center {i}"] = center
#                 record[f"Coverage {i} (%)"] = coverage

#             records.append(record)

#         except Exception as e:
#             print(f"Error parsing block:\n{block}\nError: {e}")

#     return pd.DataFrame(records)

# if __name__ == "__main__":
#     df = parse_log_file("kmeans_keq5output.txt")  # Replace with your actual path
#     df.to_excel("kmeans_output_kequals5.xlsx", index=False)
#     print("Saved data ")

import re
import pandas as pd

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = content.strip().split("Processing tifs corresponding to:")

    records = []

    for block in blocks[1:]:
        lines = block.strip().splitlines()
        try:
            msavi_file = lines[0].strip()
            image_name = lines[1].strip()
            label = lines[2].strip()
            msavi_path = lines[3].replace("Loaded MSAVI from: ", "").strip()
            msavi_mean = float(lines[4].strip())

            # Extract cluster centers
            center_line = lines[5]
            centers = re.findall(r"[-]?\d+\.\d+", center_line)
            centers = list(map(float, centers))

            # Extract coverages from subsequent lines (until "Moran's I")
            cluster_info = []
            idx = 6
            while "Cluster" in lines[idx]:
                center_match = re.search(r"center=([-]?\d+\.\d+)", lines[idx])
                coverage_match = re.search(r"coverage: ([\d.]+) %", lines[idx])
                if center_match and coverage_match:
                    center = float(center_match.group(1))
                    coverage = float(coverage_match.group(1))
                    cluster_info.append((center, coverage))
                idx += 1

            # Sort clusters by intensity
            cluster_info.sort(key=lambda x: x[0])

            # Weed cluster = highest center
            weed_cluster = cluster_info[-1]
            weed_value = weed_cluster[0]
            weed_coverage = weed_cluster[1]

            # Moran's I and p-value
            moran_line = lines[idx]
            moran_i, p_value = map(float, re.findall(r"[-]?\d+\.\d+", moran_line))

            # Plot path
            plot_path = lines[idx + 3].replace("Plot saved to: ", "").strip()

            # Construct row
            record = {
                "Image Name": image_name,
                "Label": label,
                "MSAVI Path": msavi_path,
                "Mean MSAVI": msavi_mean,
                "Moran's I": moran_i,
                "p-value": p_value,
                "Weed Cluster Value": weed_value,
                "Weed Coverage (%)": weed_coverage,
                "Plot Path": plot_path
            }

            # Add sorted cluster info
            for i, (center, coverage) in enumerate(cluster_info):
                record[f"Cluster Center {i}"] = center
                record[f"Coverage {i} (%)"] = coverage

            records.append(record)

        except Exception as e:
            print(f"Error parsing block:\n{block}\nError: {e}")

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = parse_log_file("kmeans_keq5output.txt")  # Replace with your actual file
    df.to_excel("kmeans_keq5output_.xlsx", index=False)
    print("Saved data to weed_analysis_summary_k5.xlsx")
