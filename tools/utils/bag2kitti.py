# pip install bagpy

import typer
import bagpy
import numpy as np
from pathlib import Path

data_str_map = {1: 'INT8', 2: 'UINT8', 3: 'INT16', 4: 'UINT16', 5: 'INT32', 6: 'UINT32', 7: 'FLOAT32', 8: 'FLOAT64'}
data_fmt_map = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def select_topic(bag):
    topics = list(bag.topics)
    num_topics = len(topics)

    print_bag_infos(bag)

    points_topic_indices = typer.prompt(f"select topic ({0}~{num_topics - 1})", type=int)
    assert 0 <= points_topic_indices < num_topics
    selected_topic = topics[points_topic_indices]
    print(f"{selected_topic} is selected.")
    return selected_topic


def print_bag_infos(bag):
    from prettytable import PrettyTable, SINGLE_BORDER
    table = PrettyTable(field_names=['#', 'name', 'type', 'count', 'connections', 'freq'], float_format=".2")
    table.set_style(SINGLE_BORDER)
    table.add_rows([(i, k, *v) for i, (k, v) in enumerate(zip(bag.topics, bag.topic_tuple))])
    print(table)


def select_field(msg):
    num_fields = len(msg.fields)
    print_field_infos(msg)
    intensity_indices = typer.prompt(f"select intensity filed ({0}~{num_fields - 1})", type=int)
    fields = [msg.fields[i] for i in [0, 1, 2, intensity_indices]]
    np_type = np.dtype(dict(
        names=['x', 'y', 'z', 'intensity'],
        formats=[data_fmt_map[f.datatype] for f in fields],
        offsets=[f.offset for f in fields],
        itemsize=msg.point_step
    ))
    return np_type


def print_field_infos(msg):
    from prettytable import PrettyTable, SINGLE_BORDER

    table = PrettyTable(title="fields", field_names=['#', 'name', 'offset', 'datatype', 'count'], float_format=".2")
    table.set_style(SINGLE_BORDER)
    table.add_rows([(i, f.name, f.offset, data_str_map[f.datatype], f.count) for i, f in enumerate(msg.fields)])
    print(table)


def run_lidar_odom(data_path):
    import os
    try:
        import kiss_icp
    except ImportError as e:
        os.system("pip install kiss-icp")
    sequence_path = data_path.parent
    sequence_name = data_path.stem
    _ = os.system(f"cd {str(sequence_path)} && kiss_icp_pipeline {str(sequence_name)}")
    result_path = sequence_path / 'results'
    odom_path = result_path / 'latest/pointclouds_poses.npy'
    save_path = sequence_path / 'pointclouds_poses.npy'
    os.system(f"mv {str(odom_path)} {str(save_path)}")
    os.system(f"rm -rf {str(result_path)}")


def app(file: Path = typer.Option("", "--bag", prompt=True),
        save: Path = typer.Option("."),
        max_len: int = typer.Option(200),
        intensity: int = typer.Option(0)):
    def save_subset(subseq_id):
        subset_root = (save / f'{file.stem}_{subseq_id}').resolve()
        points_dir = subset_root / 'pointclouds'
        timestamps_file = subset_root / 'timestamps.txt'
        points_dir.mkdir(exist_ok=False, parents=True)

        for i, pts in enumerate(points):
            pts.tofile(open(points_dir / f'{i % max_len:09d}.bin', 'wb'))
        with open(timestamps_file, 'w') as f:
            f.write('\n'.join(timestamps))

        print("run lidar odom ...")
        run_lidar_odom(points_dir)

        points.clear()
        timestamps.clear()

    bag = bagpy.bagreader(str(file), verbose=False)
    selected_topic = select_topic(bag)

    np_type = None
    num_subset = 0
    points = []
    timestamps = []
    for msg_id, (topic, msg, t) in enumerate(bag.reader.read_messages(topics=selected_topic)):
        if msg_id == 0:
            np_type = select_field(msg)
            print(f"dtype: {np_type}")

        pc = np.frombuffer(msg.data, dtype=np_type)
        pc = np.hstack([pc[field][..., None] for field in np_type.names]).astype(np.float32)
        if intensity:
            pc[:, -1] /= (np.max(pc[:, -1]) if intensity == -1 else intensity)
        elif intensity:
            pc[:, -1] = 0

        points.append(pc)
        timestamps.append(str(msg.header.stamp))

        if (msg_id + 1) % max_len == 0:
            num_subset += 1
            save_subset(num_subset)
    if points:
        num_subset += 1
        save_subset(num_subset)


typer.run(app)
