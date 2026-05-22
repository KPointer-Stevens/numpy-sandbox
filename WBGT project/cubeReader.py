import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

COLORMAPS = {
	"coolwarm": mpl.colors.ListedColormap(mpl.colormaps['coolwarm'](np.linspace(0.0, 1.0, 256))),
	"RdYlGn": mpl.colors.ListedColormap(mpl.colormaps['RdYlGn'](np.linspace(1.0, 0.0, 256))),
	"hot": mpl.colormaps["hot"],
	"viridis": mpl.colormaps["viridis"],
}


class CubeReaderApp:
	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("Numpy Matrix Slice Viewer")
		self.root.geometry("1200x750")

		self.data = None
		self.dim_names = []
		self.dim_units = []
		self.dim_mins = []
		self.dim_steps = []
		self.dim_default_indices = []
		self.name_vars = []
		self.unit_vars = []
		self.min_vars = []
		self.step_vars = []
		self.default_index_vars = []

		self.x_dim_var = tk.StringVar()
		self.y_dim_var = tk.StringVar()
		self.z_dim_var = tk.StringVar()
		self.z_index_var = tk.IntVar(value=0)
		self.default_mode_var = tk.StringVar(value="specified")
		self.colormap_var = tk.StringVar(value="coolwarm")
		self._heatmap_update_job = None

		self._build_layout()
		self._bind_auto_refresh_vars()

	def _build_layout(self):
		self.root.grid_rowconfigure(0, weight=4)
		self.root.grid_rowconfigure(1, weight=2)
		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_columnconfigure(1, weight=6)
		self.root.grid_columnconfigure(2, weight=1)

		self.top_left = ttk.LabelFrame(self.root, text="Axis Selection")
		self.top_left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

		self.top_center = ttk.LabelFrame(self.root, text="Heatmap")
		self.top_center.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

		self.top_right = ttk.LabelFrame(self.root, text="Z Slice")
		self.top_right.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)

		self.bottom_left = ttk.LabelFrame(self.root, text="Load / Matrix Info")
		self.bottom_left.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

		self.bottom_center = ttk.LabelFrame(self.root, text="Labels, Units, and Axis Ticks")
		self.bottom_center.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

		self.bottom_right = ttk.LabelFrame(self.root, text="Colormap")
		self.bottom_right.grid(row=1, column=2, sticky="nsew", padx=8, pady=8)

		self._build_axis_controls()
		self._build_heatmap_panel()
		self._build_slider_panel()
		self._build_load_info_panel()
		self._build_dimension_meta_panel()
		self._build_colormap_panel()

	def _build_axis_controls(self):
		for i in range(3):
			self.top_left.grid_rowconfigure(i, weight=0)
		self.top_left.grid_columnconfigure(1, weight=1)

		ttk.Label(self.top_left, text="X axis").grid(row=0, column=0, sticky="w", padx=6, pady=6)
		ttk.Label(self.top_left, text="Y axis").grid(row=1, column=0, sticky="w", padx=6, pady=6)
		ttk.Label(self.top_left, text="Z axis").grid(row=2, column=0, sticky="w", padx=6, pady=6)

		self.x_combo = ttk.Combobox(self.top_left, textvariable=self.x_dim_var, state="readonly", width=12)
		self.y_combo = ttk.Combobox(self.top_left, textvariable=self.y_dim_var, state="readonly", width=12)
		self.z_combo = ttk.Combobox(self.top_left, textvariable=self.z_dim_var, state="readonly", width=12)

		self.x_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
		self.y_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
		self.z_combo.grid(row=2, column=1, sticky="ew", padx=6, pady=6)

		self.x_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_heatmap_update())
		self.y_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_heatmap_update())
		self.z_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_z_dim_changed())

		default_frame = ttk.LabelFrame(self.top_left, text="Dimension defaults")
		default_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(12, 6))

		default_options = (
			("Minimum", "minimum"),
			("Maximum", "maximum"),
			("Specified Default", "specified"),
		)
		for row, (label, value) in enumerate(default_options):
			ttk.Radiobutton(
				default_frame,
				text=label,
				variable=self.default_mode_var,
				value=value,
				command=self._schedule_heatmap_update,
			).grid(row=row, column=0, sticky="w", padx=6, pady=3)

	def _build_heatmap_panel(self):
		self.figure = Figure(figsize=(6, 5), dpi=100)
		self.ax = self.figure.add_subplot(111)
		self.ax.set_title("Load a .npy matrix to begin")
		self.ax.set_xlabel("X")
		self.ax.set_ylabel("Y")
		self.im = None
		self.cbar = None

		self.canvas = FigureCanvasTkAgg(self.figure, master=self.top_center)
		self.canvas.get_tk_widget().pack(fill="both", expand=True)
		self.canvas.draw()

	def _build_slider_panel(self):
		self.top_right.grid_rowconfigure(0, weight=1)
		self.top_right.grid_columnconfigure(0, weight=1)

		self.slider_label = ttk.Label(self.top_right, text="Z value: 0")
		self.slider_label.pack(padx=8, pady=(10, 4), anchor="center")

		self.z_slider = ttk.Scale(
			self.top_right,
			orient="vertical",
			from_=0,
			to=0,
			command=self.on_slider_move,
		)
		self.z_slider.pack(fill="y", expand=True, padx=12, pady=8)

	def _build_load_info_panel(self):
		self.bottom_left.grid_columnconfigure(0, weight=1)

		self.load_btn = ttk.Button(self.bottom_left, text="Load .npy Matrix", command=self.load_matrix)
		self.load_btn.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 8))

		self.info_text = tk.Text(self.bottom_left, width=24, height=8, wrap="word")
		self.info_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 10))
		self.info_text.insert("1.0", "No matrix loaded.")
		self.info_text.config(state="disabled")

		self.bottom_left.grid_rowconfigure(1, weight=1)

	def _build_dimension_meta_panel(self):
		self.bottom_center.grid_rowconfigure(0, weight=1)
		self.bottom_center.grid_columnconfigure(0, weight=1)

		self.meta_canvas = tk.Canvas(self.bottom_center, highlightthickness=0)
		self.meta_scroll = ttk.Scrollbar(self.bottom_center, orient="vertical", command=self.meta_canvas.yview)
		self.meta_inner = ttk.Frame(self.meta_canvas)

		self.meta_inner.bind(
			"<Configure>",
			lambda _e: self.meta_canvas.configure(scrollregion=self.meta_canvas.bbox("all")),
		)

		self.meta_canvas.create_window((0, 0), window=self.meta_inner, anchor="nw")
		self.meta_canvas.configure(yscrollcommand=self.meta_scroll.set)

		self.meta_canvas.grid(row=0, column=0, sticky="nsew")
		self.meta_scroll.grid(row=0, column=1, sticky="ns")

	def _build_colormap_panel(self):
		self.bottom_right.grid_columnconfigure(0, weight=1)
		for row, label in enumerate(COLORMAPS):
			ttk.Radiobutton(
				self.bottom_right,
				text=label,
				variable=self.colormap_var,
				value=label,
				command=self._schedule_heatmap_update,
			).grid(row=row, column=0, sticky="w", padx=8, pady=(10 if row == 0 else 4, 4))

	def _bind_auto_refresh_vars(self):
		self.x_dim_var.trace_add("write", lambda *_args: self._schedule_heatmap_update())
		self.y_dim_var.trace_add("write", lambda *_args: self._schedule_heatmap_update())
		self.z_dim_var.trace_add("write", lambda *_args: self.on_z_dim_changed())
		self.z_index_var.trace_add("write", lambda *_args: self._schedule_heatmap_update())

	def _schedule_heatmap_update(self):
		if self._heatmap_update_job is not None:
			return
		self._heatmap_update_job = self.root.after_idle(self._run_scheduled_heatmap_update)

	def _run_scheduled_heatmap_update(self):
		self._heatmap_update_job = None
		self.update_heatmap()

	def _remove_colorbar(self):
		if self.cbar is None:
			return
		try:
			self.cbar.remove()
		except (AttributeError, KeyError, ValueError):
			pass
		self.cbar = None

	def _refresh_meta_entries(self):
		for child in self.meta_inner.winfo_children():
			child.destroy()

		self.name_vars = []
		self.unit_vars = []
		self.min_vars = []
		self.step_vars = []
		self.default_index_vars = []

		if self.data is None:
			ttk.Label(self.meta_inner, text="Load a matrix to configure dimension labels.").grid(
				row=0, column=0, padx=8, pady=8, sticky="w"
			)
			return

		header = ["Dimension", "Name", "Units", "Min", "Step", "Default Index"]
		for c, h in enumerate(header):
			ttk.Label(self.meta_inner, text=h).grid(row=0, column=c, padx=6, pady=6, sticky="w")

		for d in range(self.data.ndim):
			name_var = tk.StringVar(value=self.dim_names[d])
			unit_var = tk.StringVar(value=self.dim_units[d])
			min_var = tk.StringVar(value=self.dim_mins[d])
			step_var = tk.StringVar(value=self.dim_steps[d])
			default_index_var = tk.StringVar(value=self.dim_default_indices[d])

			self.name_vars.append(name_var)
			self.unit_vars.append(unit_var)
			self.min_vars.append(min_var)
			self.step_vars.append(step_var)
			self.default_index_vars.append(default_index_var)

			ttk.Label(self.meta_inner, text=f"dim {d} (size={self.data.shape[d]})").grid(
				row=d + 1, column=0, padx=6, pady=4, sticky="w"
			)

			name_entry = ttk.Entry(self.meta_inner, textvariable=name_var, width=14)
			unit_entry = ttk.Entry(self.meta_inner, textvariable=unit_var, width=8)
			min_entry = ttk.Entry(self.meta_inner, textvariable=min_var, width=8)
			step_entry = ttk.Entry(self.meta_inner, textvariable=step_var, width=8)
			default_index_entry = ttk.Entry(self.meta_inner, textvariable=default_index_var, width=10)
			name_entry.grid(row=d + 1, column=1, padx=6, pady=4, sticky="ew")
			unit_entry.grid(row=d + 1, column=2, padx=6, pady=4, sticky="ew")
			min_entry.grid(row=d + 1, column=3, padx=6, pady=4, sticky="ew")
			step_entry.grid(row=d + 1, column=4, padx=6, pady=4, sticky="ew")
			default_index_entry.grid(row=d + 1, column=5, padx=6, pady=4, sticky="ew")

			name_var.trace_add("write", lambda *_args, idx=d: self._on_meta_changed(idx))
			unit_var.trace_add("write", lambda *_args, idx=d: self._on_meta_changed(idx))
			min_var.trace_add("write", lambda *_args, idx=d: self._on_meta_changed(idx))
			step_var.trace_add("write", lambda *_args, idx=d: self._on_meta_changed(idx))
			default_index_var.trace_add("write", lambda *_args, idx=d: self._on_meta_changed(idx))

		self.meta_inner.grid_columnconfigure(1, weight=1)

	def _on_meta_changed(self, idx: int):
		if self.data is None:
			return
		self.dim_names[idx] = self.name_vars[idx].get().strip() or f"dim {idx}"
		self.dim_units[idx] = self.unit_vars[idx].get().strip()
		self.dim_mins[idx] = self.min_vars[idx].get().strip()
		self.dim_steps[idx] = self.step_vars[idx].get().strip()
		self.dim_default_indices[idx] = self.default_index_vars[idx].get().strip()
		self._update_dropdown_labels()
		self._update_z_slider_label()
		self._schedule_heatmap_update()

	def _update_dropdown_labels(self):
		if self.data is None:
			return
		selected_dims = (
			self._parse_dim_index(self.x_dim_var.get()),
			self._parse_dim_index(self.y_dim_var.get()),
			self._parse_dim_index(self.z_dim_var.get()),
		)
		labels = [self._dim_label(i) for i in range(self.data.ndim)]
		self.x_combo["values"] = labels
		self.y_combo["values"] = labels
		self.z_combo["values"] = labels
		for var, dim_idx in zip((self.x_dim_var, self.y_dim_var, self.z_dim_var), selected_dims):
			if dim_idx is not None and 0 <= dim_idx < len(labels):
				var.set(labels[dim_idx])

	def _dim_label(self, idx: int) -> str:
		name = self.dim_names[idx] if idx < len(self.dim_names) else f"dim {idx}"
		return f"{idx}: {name}"

	@staticmethod
	def _parse_dim_index(label: str):
		if not label:
			return None
		try:
			return int(label.split(":", 1)[0].strip())
		except Exception:
			return None

	def _set_default_dimension_meta(self, ndim: int):
		self.dim_names = [f"dim {i}" for i in range(ndim)]
		self.dim_units = ["" for _ in range(ndim)]
		self.dim_mins = ["0" for _ in range(ndim)]
		self.dim_steps = ["1" for _ in range(ndim)]
		self.dim_default_indices = [str((self.data.shape[i] - 1) // 2) for i in range(ndim)]

	def _key_path_for_matrix(self, matrix_path: str) -> Path:
		path = Path(matrix_path)
		return path.with_name(f"{path.stem}_key{path.suffix}")

	def _load_dimension_key(self, matrix_path: str):
		key_path = self._key_path_for_matrix(matrix_path)
		if not key_path.exists():
			return key_path, False

		try:
			key_data = np.load(key_path, allow_pickle=False)
		except Exception as exc:
			messagebox.showwarning("Key Load Error", f"Could not load key file:\n{key_path}\n\n{exc}")
			return key_path, False

		key_data = np.asarray(key_data)
		if key_data.ndim != 2 or key_data.shape[1] < 4:
			messagebox.showwarning(
				"Invalid Key Matrix",
				f"Key file must have rows formatted as: name, unit, step, min\n\n{key_path}",
			)
			return key_path, False

		for dim_idx in range(min(self.data.ndim, key_data.shape[0])):
			row = key_data[dim_idx]
			name = str(row[0]).strip()
			unit = str(row[1]).strip()
			step = str(row[2]).strip()
			min_value = str(row[3]).strip()

			self.dim_names[dim_idx] = name or f"dim {dim_idx}"
			self.dim_units[dim_idx] = unit
			self.dim_steps[dim_idx] = step or "1"
			self.dim_mins[dim_idx] = min_value or "0"

		return key_path, True

	def load_matrix(self):
		path = filedialog.askopenfilename(
			title="Select numpy matrix",
			filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
		)
		if not path:
			return

		try:
			arr = np.load(path, allow_pickle=False)
		except Exception as exc:
			messagebox.showerror("Load Error", f"Could not load file:\n{exc}")
			return

		if not isinstance(arr, np.ndarray):
			messagebox.showerror("Invalid Data", "Loaded object is not a numpy ndarray.")
			return
		if arr.ndim < 3:
			messagebox.showerror("Invalid Matrix", "Matrix must have 3 or more dimensions.")
			return
		if not np.issubdtype(arr.dtype, np.number):
			messagebox.showerror("Invalid Matrix", "Matrix values must be numerical.")
			return

		self.data = arr
		self._set_default_dimension_meta(arr.ndim)
		key_path, key_loaded = self._load_dimension_key(path)

		labels = [self._dim_label(i) for i in range(arr.ndim)]
		self.x_combo["values"] = labels
		self.y_combo["values"] = labels
		self.z_combo["values"] = labels

		self.x_dim_var.set(labels[0])
		self.y_dim_var.set(labels[1])
		self.z_dim_var.set(labels[2])

		self.on_z_dim_changed()
		self._refresh_meta_entries()
		self._update_info(path, key_path, key_loaded)
		self.update_heatmap()

	def _update_info(self, path: str, key_path=None, key_loaded=False):
		if self.data is None:
			return
		key_text = ""
		if key_path is not None:
			key_text = f"Key file: {key_path if key_loaded else f'not loaded ({key_path})'}\n"
		text = (
			f"File: {path}\n"
			f"{key_text}"
			f"Shape: {self.data.shape}\n"
			f"Dimensions: {self.data.ndim}\n"
			f"Total values: {self.data.size}\n"
			f"Dtype: {self.data.dtype}\n"
			"\n"
			"Note: Dimensions other than X/Y/Z use the selected dimension default mode."
		)
		self.info_text.config(state="normal")
		self.info_text.delete("1.0", "end")
		self.info_text.insert("1.0", text)
		self.info_text.config(state="disabled")

	def on_z_dim_changed(self):
		if self.data is None:
			return
		z_dim = self._parse_dim_index(self.z_dim_var.get())
		if z_dim is None:
			return

		max_idx = max(0, self.data.shape[z_dim] - 1)
		self.z_slider.configure(from_=max_idx, to=0)
		if self.z_index_var.get() > max_idx:
			self.z_index_var.set(max_idx)
		self.z_slider.set(self.z_index_var.get())
		self._update_z_slider_label()
		self._schedule_heatmap_update()

	def on_slider_move(self, val):
		try:
			idx = int(round(float(val)))
		except Exception:
			idx = 0
		self.z_index_var.set(idx)
		self._update_z_slider_label()
		self._schedule_heatmap_update()

	def _axis_title(self, dim_idx: int):
		name = self.dim_names[dim_idx]
		unit = self.dim_units[dim_idx].strip()
		return f"{name} [{unit}]" if unit else name

	def _selected_axis_color_limits(self, x_dim: int, y_dim: int, z_dim: int):
		idx = self._base_dimension_indices()
		idx[x_dim] = slice(None)
		idx[y_dim] = slice(None)
		idx[z_dim] = slice(None)

		selected_data = np.asarray(self.data[tuple(idx)])
		finite_values = selected_data[np.isfinite(selected_data)]
		if finite_values.size == 0:
			return None, None

		vmin = float(np.min(finite_values))
		vmax = float(np.max(finite_values))
		if vmin == vmax:
			padding = abs(vmin) * 0.01 or 0.5
			vmin -= padding
			vmax += padding
		return vmin, vmax

	@staticmethod
	def _parse_float(value: str):
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	def _dim_tick_values(self, dim_idx: int):
		size = self.data.shape[dim_idx]
		min_value = self._parse_float(self.dim_mins[dim_idx])
		step_value = self._parse_float(self.dim_steps[dim_idx])

		if min_value is not None and step_value is not None:
			return min_value + step_value * np.arange(size)
		return np.arange(size)

	def _format_tick_value(self, value):
		return f"{value:.6g}"

	def _parse_index(self, value: str):
		try:
			return int(float(value))
		except (TypeError, ValueError):
			return 0

	def _clamp_index(self, dim_idx: int, index: int):
		return min(max(index, 0), self.data.shape[dim_idx] - 1)

	def _dimension_default_index(self, dim_idx: int):
		mode = self.default_mode_var.get()
		if mode == "minimum":
			return 0
		if mode == "maximum":
			return self.data.shape[dim_idx] - 1
		if dim_idx < len(self.dim_default_indices):
			return self._clamp_index(dim_idx, self._parse_index(self.dim_default_indices[dim_idx]))
		return 0

	def _base_dimension_indices(self):
		return [self._dimension_default_index(dim_idx) for dim_idx in range(self.data.ndim)]

	def _update_z_slider_label(self):
		if self.data is None:
			self.slider_label.config(text="Z value: 0")
			return

		z_dim = self._parse_dim_index(self.z_dim_var.get())
		if z_dim is None:
			self.slider_label.config(text="Z value: 0")
			return

		max_idx = max(0, self.data.shape[z_dim] - 1)
		idx = min(max(self.z_index_var.get(), 0), max_idx)
		z_values = self._dim_tick_values(z_dim)
		self.slider_label.config(text=f"{self._axis_title(z_dim)}: {self._format_tick_value(z_values[idx])}")

	def _apply_axis_ticks(self, x_dim: int, y_dim: int):
		for dim_idx, setter, label_setter in (
			(x_dim, self.ax.set_xticks, self.ax.set_xticklabels),
			(y_dim, self.ax.set_yticks, self.ax.set_yticklabels),
		):
			size = self.data.shape[dim_idx]
			if size <= 20:
				tick_positions = np.arange(size)
			else:
				tick_count = min(12, size)
				tick_positions = np.linspace(0, size - 1, tick_count, dtype=int)
				tick_positions = np.unique(tick_positions)
			tick_values = self._dim_tick_values(dim_idx)

			setter(tick_positions)
			label_setter([self._format_tick_value(tick_values[i]) for i in tick_positions])

	def _orient_slice_for_axes(self, slice_data, x_dim: int, y_dim: int):
		remaining_dims = [dim for dim in range(self.data.ndim) if dim in (x_dim, y_dim)]
		y_axis = remaining_dims.index(y_dim)
		x_axis = remaining_dims.index(x_dim)
		return np.moveaxis(slice_data, (y_axis, x_axis), (0, 1))

	def update_heatmap(self):
		if self.data is None:
			return

		x_dim = self._parse_dim_index(self.x_dim_var.get())
		y_dim = self._parse_dim_index(self.y_dim_var.get())
		z_dim = self._parse_dim_index(self.z_dim_var.get())
		if x_dim is None or y_dim is None or z_dim is None:
			return

		if len({x_dim, y_dim, z_dim}) < 3:
			self._remove_colorbar()
			self.ax.clear()
			self.ax.text(0.5, 0.5, "X, Y, and Z must be different dimensions", ha="center", va="center")
			self.ax.set_axis_off()
			self.canvas.draw_idle()
			return

		idx = self._base_dimension_indices()
		idx[z_dim] = min(self.z_index_var.get(), self.data.shape[z_dim] - 1)
		idx[x_dim] = slice(None)
		idx[y_dim] = slice(None)

		slice_data = self.data[tuple(idx)]
		if slice_data.ndim != 2:
			self._remove_colorbar()
			self.ax.clear()
			self.ax.text(0.5, 0.5, "Unable to derive 2D slice", ha="center", va="center")
			self.ax.set_axis_off()
			self.canvas.draw_idle()
			return

		self.ax.clear()
		self.ax.set_axis_on()
		slice_data = self._orient_slice_for_axes(slice_data, x_dim, y_dim)
		vmin, vmax = self._selected_axis_color_limits(x_dim, y_dim, z_dim)
		self.im = self.ax.imshow(
			np.asarray(slice_data),
			origin="lower",
			aspect="auto",
			cmap=COLORMAPS[self.colormap_var.get()],
			vmin=vmin,
			vmax=vmax,
		)
		self.ax.set_box_aspect(1)

		self.ax.set_xlabel(self._axis_title(x_dim))
		self.ax.set_ylabel(self._axis_title(y_dim))
		self.ax.set_title(f"Slice at {self._axis_title(z_dim)} index {idx[z_dim]}")
		self._apply_axis_ticks(x_dim, y_dim)

		if self.cbar is None:
			self.cbar = self.figure.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
		else:
			self.cbar.update_normal(self.im)
		self.cbar.set_label("Value")

		self.canvas.draw_idle()


def main():
	root = tk.Tk()
	app = CubeReaderApp(root)
	app._refresh_meta_entries()
	root.mainloop()


if __name__ == "__main__":
	main()
