"""
load_spe.py

Python port of LoadSPE_F1.m -- read Princeton Instruments .spe spectral
data files, supporting both:
  - LightField SPE v3.x  (XML footer describes the data layout)
  - WinSpec   SPE v2.x   (fixed-offset binary header, e.g. v2.2)

Usage
-----
    from load_spe import load_spe
    data, wavelengths, params = load_spe("yourfile.spe")

Returns
-------
data : numpy.ndarray
    - Single ROI, single frame -> 2D array, shape (width, height)
    - Single ROI, multiple frames -> 3D array, shape (width, height, n_frames)
    - Multiple ROIs, single frame (v3.x only) -> dict {'ROI': [array, ...]}
    - Multiple ROIs, multiple frames (v3.x only) -> list of dicts, one per frame

wavelengths : numpy.ndarray (or list of arrays / dict for multi-ROI v3.x files)
    Wavelength axis in nm, derived from the calibration stored in the file.
    Falls back to a plain pixel index if no valid calibration is present.

params : dict
    Assorted metadata pulled from the header/XML footer. For v2.x files this
    includes params['full'], a dict of every header field. For v3.x files
    params['SpeFormat'] holds the parsed XML root element.

Ported from M. Sich's MATLAB `loadSPE` (v2.7, 2018), University of Sheffield,
itself based on the Princeton Instruments SPE 2.x/3.0 file format specs.
"""

import struct
import warnings
import xml.etree.ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------
# Small sequential binary reader, mirroring MATLAB's fread/fseek semantics
# --------------------------------------------------------------------------
class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def seek(self, pos):
        self.pos = pos

    def skip(self, n):
        self.pos += n

    def _read(self, fmt, size):
        val = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return val

    def i16(self):
        return self._read('<h', 2)[0]

    def u16(self):
        return self._read('<H', 2)[0]

    def i32(self):
        return self._read('<i', 4)[0]

    def u32(self):
        return self._read('<I', 4)[0]

    def f32(self):
        return self._read('<f', 4)[0]

    def f64(self):
        return self._read('<d', 8)[0]

    def u8(self):
        return self._read('<B', 1)[0]

    def char(self, n):
        raw = self.data[self.pos:self.pos + n]
        self.pos += n
        # Fixed-width, null-padded field -> stripped string
        return raw.split(b'\x00', 1)[0].decode('latin-1', errors='replace').strip()

    def i16_arr(self, n):
        vals = struct.unpack_from(f'<{n}h', self.data, self.pos)
        self.pos += 2 * n
        return list(vals)

    def f32_arr(self, n):
        vals = struct.unpack_from(f'<{n}f', self.data, self.pos)
        self.pos += 4 * n
        return list(vals)

    def f64_arr(self, n):
        vals = struct.unpack_from(f'<{n}d', self.data, self.pos)
        self.pos += 8 * n
        return list(vals)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def load_spe(filename):
    """Load a Princeton Instruments .spe file (v2.x or v3.x)."""
    with open(filename, 'rb') as fid:
        raw = fid.read()

    params = {}

    # SPE version lives at byte offset 1992 as a 4-byte float, for both
    # v2.x and v3.x files.
    version = struct.unpack_from('<f', raw, 1992)[0]
    params['version'] = version

    if version >= 3:
        return _load_v3(raw, params, filename)
    elif version >= 2:
        return _load_v2(raw, params, filename)
    else:
        raise ValueError(f"Unsupported version of SPE file: {version}")


# --------------------------------------------------------------------------
# SPE v3.x -- XML footer describes everything
# --------------------------------------------------------------------------
def _load_v3(raw, params, filename):
    footer_offset = struct.unpack_from('<Q', raw, 678)[0]
    xml_text = raw[footer_offset:].decode('utf-8', errors='replace')

    start = xml_text.find('<SpeFormat')
    if start == -1:
        raise ValueError(f"Could not find SpeFormat XML footer in {filename}")
    # Trim to the matching closing tag so any trailing junk doesn't break the parser
    end = xml_text.rfind('</SpeFormat>')
    root = ET.fromstring(xml_text[start:end + len('</SpeFormat>')] if end != -1 else xml_text[start:])

    ns = {'s': root.tag.split('}')[0].strip('{')} if root.tag.startswith('{') else None

    def find(elem, path):
        if ns:
            path = '/'.join(f's:{p}' for p in path.split('/'))
            return elem.find(path, ns)
        return elem.find(path)

    def findall(elem, path):
        if ns:
            path = '/'.join(f's:{p}' for p in path.split('/'))
            return elem.findall(path, ns)
        return elem.findall(path)

    params['SpeFormat'] = root

    data_block = find(root, 'DataFormat/DataBlock')
    pixel_format = data_block.get('pixelFormat')
    dtype_map = {
        'MonochromeUnsigned16': np.uint16,
        'MonochromeUnsigned32': np.uint32,
        'MonochromeFloating32': np.float32,
    }
    if pixel_format not in dtype_map:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")
    dtype = dtype_map[pixel_format]
    itemsize = np.dtype(dtype).itemsize

    n_frames = int(data_block.get('count'))
    roi_blocks = findall(data_block, 'DataBlock')
    n_roi = len(roi_blocks) if roi_blocks else 1

    offset = 4100  # data always starts at byte 4100, as in v2.x

    def read_block(blk, off):
        width = int(blk.get('width'))
        height = int(blk.get('height'))
        n = width * height
        arr = np.frombuffer(raw, dtype=dtype, count=n, offset=off)
        # MATLAB fread([width, height]) fills column-major -> transpose to match
        return arr.reshape((height, width)).T

    if n_roi == 1:
        blk = roi_blocks[0] if roi_blocks else data_block
        width, height = int(blk.get('width')), int(blk.get('height'))
        frame_size = width * height
        if n_frames == 1:
            data = read_block(blk, offset)
        else:
            frames = [read_block(blk, offset + j * frame_size * itemsize) for j in range(n_frames)]
            data = np.stack(frames, axis=2)
    else:
        if n_frames == 1:
            data = {'ROI': []}
            for blk in roi_blocks:
                data['ROI'].append(read_block(blk, offset))
                offset += int(blk.get('width')) * int(blk.get('height')) * itemsize
        else:
            data = []
            for j in range(n_frames):
                frame = {'ROI': []}
                for blk in roi_blocks:
                    frame['ROI'].append(read_block(blk, offset))
                    offset += int(blk.get('width')) * int(blk.get('height')) * itemsize
                data.append(frame)

    # Wavelength calibration
    wavelength_mapping = find(root, 'Calibrations/WavelengthMapping/Wavelength')
    sensor_mappings = findall(root, 'Calibrations/SensorMapping')

    if wavelength_mapping is not None and wavelength_mapping.text:
        # Values are comma-separated (occasionally with surrounding whitespace
        # too), so split on commas rather than plain whitespace.
        w = np.array([float(x) for x in wavelength_mapping.text.replace(',', ' ').split()])
        if sensor_mappings:
            if n_roi == 1:
                sm = sensor_mappings[0]
                x1 = int(sm.get('x'))
                width = int(sm.get('width'))
                wavelengths = w[x1:x1 + width]
            else:
                wavelengths = []
                for sm in sensor_mappings:
                    x1 = int(sm.get('x'))
                    width = int(sm.get('width'))
                    wavelengths.append(w[x1:x1 + width])
        else:
            # No SensorMapping present (e.g. stitched files) -- use as-is
            wavelengths = w
    else:
        wavelengths = np.array([])

    return data, wavelengths, params


# --------------------------------------------------------------------------
# SPE v2.x -- fixed binary header (e.g. v2.2 WinSpec files)
# --------------------------------------------------------------------------
def _load_v2(raw, params, filename):
    r = _Reader(raw)
    p = {}

    r.seek(0)
    p['ControllerVersion'] = r.i16()
    p['LogicOutput'] = r.i16()
    p['AmpHiCapLowNoise'] = r.u16()
    p['xDimDet'] = r.u16()
    p['mode'] = r.i16()
    p['exp_sec'] = r.f32()
    p['VChipXdim'] = r.i16()
    p['VChipYdim'] = r.i16()
    p['yDimDet'] = r.u16()
    p['date'] = r.char(10)
    p['VirtualChipFlag'] = r.i16()
    r.skip(2)  # spare chars
    p['noscan'] = r.i16()
    p['DetTemperature'] = r.f32()
    p['DetType'] = r.i16()
    p['xdim'] = r.u16()
    p['stdiode'] = r.i16()
    p['DelayTime'] = r.f32()
    p['ShutterControl'] = r.u16()
    p['AbsorbLive'] = r.i16()
    p['AbsorbMode'] = r.u16()
    p['CanDoVirtualChipFlag'] = r.i16()
    p['ThresholdMinLive'] = r.i16()
    p['ThresholdMinVal'] = r.f32()
    p['ThresholdMaxLive'] = r.i16()
    p['ThresholdMaxVal'] = r.f32()
    p['SpecAutoSpectroMode'] = r.i16()
    p['SpecCenterWlNm'] = r.f32()
    p['SpecGlueFlag'] = r.i16()
    p['SpecGlueStartWlNm'] = r.f32()
    p['SpecGlueEndWlNm'] = r.f32()
    p['SpecGlueMinOvrlpNm'] = r.f32()
    p['SpecGlueFinalResNm'] = r.f32()
    p['PulserType'] = r.i16()
    p['CustomChipFlag'] = r.i16()
    p['XPrePixels'] = r.i16()
    p['XPostPixels'] = r.i16()
    p['YPrePixels'] = r.i16()
    p['YPostPixels'] = r.i16()
    p['asynen'] = r.i16()

    datatype_code = r.i16()
    dtype_map = {0: np.float32, 1: np.int32, 2: np.int16, 3: np.uint16}
    if datatype_code not in dtype_map:
        raise ValueError(f"Unknown .SPE v2.x data type: {datatype_code}")
    p['datatype'] = dtype_map[datatype_code]

    p['PulserMode'] = r.i16()
    p['PulserOnChipAccums'] = r.u16()
    p['PulserRepeatExp'] = r.u32()
    p['PulseRepWidth'] = r.f32()
    p['PulseRepDelay'] = r.f32()
    p['PulseSeqStartWidth'] = r.f32()
    p['PulseSeqEndWidth'] = r.f32()
    p['PulseSeqStartDelay'] = r.f32()
    p['PulseSeqEndDelay'] = r.f32()
    p['PulseSeqIncMode'] = r.i16()
    p['PImaxUsed'] = r.i16()
    p['PImaxMode'] = r.i16()
    p['PImaxGain'] = r.i16()
    p['BackGrndApplied'] = r.i16()
    p['PImax2nsBrdUsed'] = r.i16()
    p['minblk'] = r.u16()
    p['numminblk'] = r.u16()
    p['SpecMirrorLocation'] = r.i16_arr(2)
    p['SpecSlitLocation'] = r.i16_arr(4)
    p['CustomTimingFlag'] = r.i16()
    p['ExperimentTimeLocal'] = r.char(7)
    p['ExperimentTimeUTC'] = r.char(7)
    p['ExposUnits'] = r.i16()
    p['ADCoffset'] = r.u16()
    p['ADCrate'] = r.u16()
    p['ADCtype'] = r.u16()
    p['ADCresolution'] = r.u16()
    p['ADCbitAdjust'] = r.u16()
    p['gain'] = r.u16()
    p['Comments'] = r.char(80)

    r.seek(600)
    p['geometric'] = r.u16()
    p['xlabel'] = r.char(16)
    p['cleans'] = r.u16()
    p['NumSkpPerCln'] = r.u16()
    p['SpecMirrorPos'] = r.i16_arr(2)
    p['SpecSlitPos'] = r.f32_arr(4)
    p['AutoCleansActive'] = r.i16()
    p['UseContCleansInst'] = r.i16()
    p['AbsorbStripNum'] = r.i16()
    p['SpecSlitPosUnits'] = r.i16()
    p['SpecGrooves'] = r.f32()
    p['srccmp'] = r.i16()
    p['ydim'] = r.u16()
    p['scramble'] = r.i16()
    p['ContinuousCleansFlag'] = r.i16()
    p['ExternalTriggerFlag'] = r.i16()
    p['lnoscan'] = r.i32()
    p['lavgexp'] = r.i32()
    p['ReadoutTime'] = r.f32()
    p['TriggeredModeFlag'] = r.i16()
    r.skip(10)  # spare chars
    p['sw_version'] = r.char(16)

    type_code = r.i16()
    controller_types = {
        1: 'new120 (Type II)', 2: 'old120 (Type I)', 3: 'ST130', 4: 'ST121',
        5: 'ST138', 6: 'DC131 (PentaMax)', 7: 'ST133 (MicroMax/SpectroMax)',
        8: 'ST135 (GPIB)', 9: 'VICCD', 10: 'ST116 (GPIB)', 11: 'OMA3 (GPIB)',
        12: 'OMA4',
    }
    p['type'] = controller_types.get(type_code, f'Unknown: {type_code}')

    p['flatFieldApplied'] = r.i16()
    r.skip(16)  # spare chars
    p['kin_trig_mode'] = r.i16()
    p['dlabel'] = r.char(16)
    r.skip(436)  # spare chars

    r.seek(1178)
    p['PulseFileName'] = r.char(120)
    p['AbsorbFileName'] = r.char(120)
    p['NumExpRepeats'] = r.u32()
    p['NumExpAccums'] = r.u32()
    p['YT_Flag'] = r.i16()
    p['clkspd_us'] = r.f32()
    p['HWaccumFlag'] = r.i16()
    p['StoreSync'] = r.i16()
    p['BlemishApplied'] = r.i16()
    p['CosmicApplied'] = r.i16()
    p['CosmicType'] = r.i16()
    p['CosmicThreshold'] = r.f32()
    p['NumFrames'] = r.i32()
    p['MaxIntensity'] = r.f32()
    p['MinIntensity'] = r.f32()
    p['ylabel'] = r.char(16)
    p['ShutterType'] = r.u16()
    p['shutterComp'] = r.f32()
    p['readoutMode'] = r.u16()
    p['WindowSize'] = r.u16()
    p['clkspd'] = r.u16()
    p['interface_type'] = r.u16()
    p['NumROIsInExperiment'] = r.i16() or 1
    r.skip(16)  # spare chars
    p['controllerNum'] = r.u16()
    p['SWmade'] = r.u16()

    num_roi = r.i16()
    if num_roi == 0:
        num_roi = 1
    p['NumROI'] = num_roi
    n = min(num_roi, 10)

    roi_offsets = [1512, 1524, 1536, 1548, 1560, 1572, 1584, 1596, 1608, 1620]
    rois = []
    for i in range(n):
        r.seek(roi_offsets[i])
        rois.append({
            'startx': r.u16(), 'endx': r.u16(), 'groupx': r.u16(),
            'starty': r.u16(), 'endy': r.u16(), 'groupy': r.u16(),
        })
    p['ROI'] = rois

    r.seek(1632)
    p['FlatField'] = r.char(120)
    p['background'] = r.char(120)
    p['blemish'] = r.char(120)
    p['file_header_ver'] = r.f32()
    p['YT_Info'] = r.char(1000)
    p['WinView_id'] = r.i32()

    # Key parameters mirroring the top-level MATLAB "params" struct
    params['ROI'] = p['ROI']
    params['xdim'] = p['xdim']
    params['ydim'] = p['ydim']
    params['xlabel'] = p['xlabel']
    params['ylabel'] = p['ylabel']
    params['dlabel'] = p['dlabel']
    params['SpecGrooves'] = p['SpecGrooves']
    params['ExperimentTimeLocal'] = p['ExperimentTimeLocal']
    params['date'] = p['date']
    params['exp_sec'] = p['exp_sec']

    # X calibration
    r.seek(3000)
    xc = {}
    xc['offset'] = r.f64()
    xc['factor'] = r.f64()
    xc['current_unit'] = r.char(1)
    r.seek(3018)
    xc['special_string'] = r.char(40)
    r.seek(3098)
    xc['calib_valid'] = r.u8()
    xc['input_unit'] = r.u8()
    xc['polynom_unit'] = r.u8()
    xc['polynom_order'] = r.u8()
    xc['calib_count'] = r.u8()
    xc['pixel_position'] = r.f64_arr(10)
    xc['calib_value'] = r.f64_arr(10)
    xc['polynom_coeff'] = r.f64_arr(6)
    xc['laser_position'] = r.f64()
    r.seek(3320)
    xc['new_calib_flag'] = r.u8()
    xc['calib_label'] = r.char(81)
    xc['expansion'] = r.char(87)
    params['xcalib'] = xc

    # Y calibration
    r.seek(3498)
    yc = {}
    yc['offset'] = r.f64()
    yc['factor'] = r.f64()
    yc['current_unit'] = r.char(1)
    r.seek(3507)
    yc['special_string'] = r.char(40)
    r.seek(3587)
    yc['calib_valid'] = r.u8()
    yc['input_unit'] = r.u8()
    yc['polynom_unit'] = r.u8()
    yc['polynom_order'] = r.u8()
    yc['calib_count'] = r.u8()
    yc['pixel_position'] = r.f64_arr(10)
    yc['calib_value'] = r.f64_arr(10)
    yc['polynom_coeff'] = r.f64_arr(6)
    yc['laser_position'] = r.f64()
    r.seek(3809)
    yc['new_calib_flag'] = r.u8()
    yc['calib_label'] = r.char(81)
    yc['expansion'] = r.char(87)
    params['ycalib'] = yc

    # Wavelength calibration -- prefer x-axis polynomial, fall back to y-axis,
    # then to plain pixel indices.
    if xc['calib_valid']:
        w_pixels = np.arange(1, p['xdim'] + 1)
        wavelengths = np.zeros(p['xdim'])
        for i in range(int(xc['polynom_order'])):
            wavelengths += xc['polynom_coeff'][i] * w_pixels ** i
    elif yc['calib_valid']:
        warnings.warn(
            f"Applying y-axis calibration to wavelengths, no x-axis "
            f"calibration found in {filename}"
        )
        w_pixels = np.arange(1, p['ydim'] + 1)
        wavelengths = np.zeros(p['ydim'])
        for i in range(int(yc['polynom_order'])):
            wavelengths += yc['polynom_coeff'][i] * w_pixels ** i
    else:
        warnings.warn(
            f"No valid x- or y-axis wavelength calibration found in "
            f"{filename}. Reverting to pixels."
        )
        wavelengths = np.arange(1, p['xdim'] + 1)

    params['full'] = p

    # Load frame data
    r.seek(4100)
    dtype = p['datatype']
    xdim, ydim = p['xdim'], p['ydim']
    n_frames = p['NumFrames'] if p['NumFrames'] not in (0,) else 1
    frame_size = xdim * ydim
    itemsize = np.dtype(dtype).itemsize

    if n_frames <= 1:
        arr = np.frombuffer(raw, dtype=dtype, count=frame_size, offset=r.pos)
        data = arr.reshape((ydim, xdim)).T  # column-major fill, like MATLAB fread
    else:
        frames = []
        for i in range(n_frames):
            off = r.pos + i * frame_size * itemsize
            arr = np.frombuffer(raw, dtype=dtype, count=frame_size, offset=off)
            frames.append(arr.reshape((ydim, xdim)).T)
        data = np.stack(frames, axis=2)

    return data, wavelengths, params


def pick_spe_file():
    """Open a native file-picker dialog and return the chosen .spe path.

    Requires Tk support (on macOS with Homebrew Python: `brew install
    python-tk@3.13`, matching your Python version). Returns None if the
    user cancels.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()          # hide the empty root window
    root.attributes('-topmost', True)  # bring the dialog to the front on macOS
    path = filedialog.askopenfilename(
        title="Select a .spe file",
        filetypes=[("SPE files", "*.spe"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


# --------------------------------------------------------------------------
# Quick demo / smoke test
# --------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = pick_spe_file()
        if not filepath:
            print("No file selected.")
            sys.exit(0)

    data, wavelengths, params = load_spe(filepath)
    print(f"SPE version: {params['version']}")
    print(f"Data shape: {getattr(data, 'shape', type(data))}")

    if isinstance(data, np.ndarray) and data.ndim == 2:
        plt.plot(wavelengths, data[:, 0])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (counts)')
        plt.title(filepath)
        plt.show()
    elif isinstance(data, np.ndarray) and data.ndim == 3:
        plt.plot(wavelengths, data[:, 0, 0])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (counts, frame 0)')
        plt.title(filepath)
        plt.show()
    else:
        print("Multi-ROI file -- inspect `data['ROI']` directly.")