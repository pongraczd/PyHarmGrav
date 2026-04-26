import sys
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QStackedWidget, QGridLayout,
    QVBoxLayout, QGroupBox, QLabel, QLineEdit, QFileDialog, QHBoxLayout, QCheckBox, QComboBox, QRadioButton, QButtonGroup
)
#from PyQt5.QtCore import QSize
from PyQt5.QtGui import QGuiApplication,QIcon
from PyQt5.QtCore import QCoreApplication
import datetime
from .pyharm_grav_shs import point_sh_synthesis, grid_sh_synthesis

def calc_point(input_file, point_numbers,output_file ,shcs_data, points_type, quantity , nmin , nmax , ellipsoid , GM , R , DTM_shcs_data , normal_field_removed):
    data_in_file = np.loadtxt(input_file)
    if point_numbers:
        points = data_in_file[:,1:]
    else:
        points = data_in_file
    if isinstance(quantity, list):
        result = []
        for quantity_item in quantity:
            result_temp = point_sh_synthesis(points ,shcs_data , points_type , quantity_item , nmin , nmax , ellipsoid , GM , R , DTM_shcs_data, normal_field_removed)
            result_temp=result_temp.reshape(-1,1)
            result.append(result_temp)
        result = np.hstack(result)
        quantity_num = len(quantity)
        if 'g' in quantity:
            quantity_num +=2
        if point_numbers:
            out_format = '%d %.8f %.8f %.3f ' + quantity_num * '%.12e '
        else:
            out_format = '%.8f %.8f %.3f ' + quantity_num * '%.12e '
        out_format = out_format.strip()
    else:
        result = point_sh_synthesis(points ,shcs_data , points_type , quantity , nmin , nmax , ellipsoid , GM , R , DTM_shcs_data, normal_field_removed)
        if result.ndim == 1:
            result = result.reshape(-1,1)
        if point_numbers:
            if quantity == 'g':
                out_format = '%d %.8f %.8f %.3f %.12e %.12e %.12e'
            else:
                out_format = '%d %.8f %.8f %.3f %.12e'
        else:
            out_format = '%.8f %.8f %.3f %.12e'
    output_array = np.hstack((data_in_file, result))
    np.savetxt(output_file,output_array,fmt=out_format)
    n_points = output_array.shape[0]
    return n_points

def calc_grid(output_file,quantity , min_lat , max_lat , min_lon , max_lon , resolution , shcs_data , resolution_unit , nmin , nmax , ellipsoid ,ref_surface_type , height ,GM , R , DTM_shcs_data , normal_field_removed):
    result ,coords = grid_sh_synthesis(quantity , min_lat , max_lat , min_lon , max_lon , resolution , shcs_data , resolution_unit , nmin , nmax , ellipsoid ,ref_surface_type , height ,GM , R , DTM_shcs_data , normal_field_removed)
    if output_file.endswith('.nc'):
        import xarray as xr
        import rioxarray
        result_ds = xr.DataArray(result,coords,name = quantity)
        result_ds.rio.write_crs(4326, inplace=True)
        result_ds.to_netcdf(output_file)
    elif output_file.endswith('.tif'):
        import xarray as xr
        import rioxarray
        result_ds = xr.DataArray(result,coords,name = quantity).astype("float32")
        result_ds.rio.write_crs(4326, inplace=True)
        result_ds.rio.to_raster(output_file)
    elif output_file.endswith('.dat') or output_file.endswith('.txt'):
        lat_grid = np.repeat((coords['latitude']).reshape(-1,1),len(coords['longitude']),axis=1)
        lon_grid = np.repeat(np.expand_dims(coords['longitude'],0),len(coords['latitude']),axis=0)
        out_array = np.vstack((lat_grid.ravel(),lon_grid.ravel(),result.ravel())).T
        np.savetxt(output_file,out_array,fmt='%.8f %.8f %.12e')
    else:
        raise ValueError('Not recognised output file type')
    
    n_lat = result.shape[0]
    n_lon = result.shape[1]
    n_points = n_lat * n_lon
    return n_lat,n_lon,n_points


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        screen = QGuiApplication.primaryScreen()
        geometry = screen.availableGeometry()
        width = int(geometry.width()*0.5) 
        height = int(geometry.height()*0.6)
        self.resize(width, height)

        #self.setFixedSize(QSize(700, 640))
        self.setWindowTitle("PyHarmGrav GUI")
        iconpath = os.path.join(os.path.dirname(__file__),"pyharmgrav.png")
        self.setWindowIcon(QIcon(iconpath))

        # Central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Panel 1: Geopotential model and reference system selection
        panel1 = QGroupBox("Geopotential model and reference system selection")
        panel1.setStyleSheet("QGroupBox { background-color: rgb(220, 220, 235); }")
        panel1_layout = QVBoxLayout()
        panel1.setLayout(panel1_layout)

        ggm_text = QLabel('Global geopotential model of the Earth')
        panel1_layout.addWidget(ggm_text)

        file_layout = QHBoxLayout()
        self.imported_ggm_file = QLineEdit()
        self.imported_ggm_file.setReadOnly(True)
        browse_input = QPushButton("... Browse")
        browse_input.setStyleSheet("QPushButton { background-color: rgb(200, 200, 255); }")
        browse_input.clicked.connect(self.open_file_dialog)
        self.use_max_GGM = QCheckBox("Use maximum degree of GGM")
        self.use_max_GGM.toggled.connect(self.toggle_nmin_input)

        # use GM and R from model
        self.use_GM_R = QCheckBox("Use GM and R of GGM")
        self.use_GM_R.setChecked(True)
        self.use_GM_R.toggled.connect(self.toggle_GM_R)

        file_layout.addWidget(browse_input)
        file_layout.addWidget(self.imported_ggm_file)
        file_layout.addWidget(self.use_max_GGM)
        file_layout.addWidget(self.use_GM_R)
        # NMIN
        GGM_additional_params = QHBoxLayout()
        nmin = QVBoxLayout()
        nmin_text = QLabel('nmin')
        nmin.addWidget(nmin_text)
        self.nmin_input = QLineEdit()
        nmin.addWidget(self.nmin_input)
        self.nmin_input.setMaximumWidth(int(width/7))
        GGM_additional_params.addLayout(nmin)
        
        # NMAX
        nmax = QVBoxLayout()
        nmax_text = QLabel('nmax')
        nmax.addWidget(nmax_text)
        self.nmax_input = QLineEdit()
        nmax.addWidget(self.nmax_input)
        self.nmax_input.setMaximumWidth(int(width/7))
        GGM_additional_params.addLayout(nmax)

        #GM
        GM = QVBoxLayout()
        GM_text = QLabel('GM')
        GM.addWidget(GM_text)
        self.GM_input = QLineEdit()
        self.GM_input.setReadOnly(True)
        self.GM_input.setStyleSheet("QLineEdit { background-color: lightgray; }")
        self.GM_input.setMaximumWidth(int(width/6))
        GM.addWidget(self.GM_input)
        GGM_additional_params.addLayout(GM)

        #R
        scale = QVBoxLayout()
        scale_text = QLabel('R')
        scale.addWidget(scale_text)
        self.scale_input = QLineEdit()
        self.scale_input.setReadOnly(True)
        self.scale_input.setStyleSheet("QLineEdit { background-color: lightgray; }")
        self.scale_input.setMaximumWidth(int(width/6))
        scale.addWidget(self.scale_input)

        GGM_additional_params.addLayout(scale)

        ellipsoid_list = ["WGS84", "GRS80"]
        ellipsoid_set = QVBoxLayout()
        ellipsoid_text = QLabel('Ellipsoid')
        ellipsoid_set.addWidget(ellipsoid_text)
        self.ell_combo_box = QComboBox()
        self.ell_combo_box.addItems(ellipsoid_list)
        #self.ell_combo_box.setMaximumWidth(int(width/6))
        ellipsoid_set.addWidget(self.ell_combo_box)
        GGM_additional_params.addLayout(ellipsoid_set)

        GGM_additional_params2 = QHBoxLayout()
        self.normal_field_removed = QCheckBox("Normal field removed from coefficients")
        SH_topo_label = QLabel('SH coefficients of topography')
        self.imported_SH_topo_file = QLineEdit()
        self.imported_SH_topo_file.setReadOnly(True)
        browse_topo_input = QPushButton("... Browse")
        browse_topo_input.setStyleSheet("QPushButton { background-color: rgb(200, 200, 255); }")
        browse_topo_input.clicked.connect(self.open_file_dialog_topo)

        GGM_additional_params2.addWidget(self.normal_field_removed)
        GGM_additional_params2.addWidget(SH_topo_label)
        GGM_additional_params2.addWidget(browse_topo_input)
        GGM_additional_params2.addWidget(self.imported_SH_topo_file)


        panel1_layout.addLayout(file_layout)
        panel1_layout.addLayout(GGM_additional_params)
        panel1_layout.addLayout(GGM_additional_params2)


        # Panel 2: Point type selection
        panel2 = QGroupBox("Point type selection")
        panel2.setStyleSheet("QGroupBox { background-color: rgb(220, 235, 220); }")
        panel2_layout = QVBoxLayout()
        panel2.setLayout(panel2_layout)

        panel2_row1 = QHBoxLayout()
        point_type_text = QLabel('Type of input coordinates')
        select_ellipsoidal = QRadioButton("Ellipsoidal")
        select_spherical = QRadioButton("Spherical")
        select_ellipsoidal.setChecked(True)  # Default selection
        panel2_row1.addWidget(point_type_text)
        panel2_row1.addWidget(select_ellipsoidal)       
        panel2_row1.addWidget(select_spherical)
        panel2_layout.addLayout(panel2_row1)

        self.button_group_sh_ell = QButtonGroup()
        self.button_group_sh_ell.addButton(select_ellipsoidal, 1)  # ID = 1
        self.button_group_sh_ell.addButton(select_spherical, 2)  # ID = 2

        panel2_row2 = QHBoxLayout()
        select_grid = QRadioButton("Grid")
        select_load_points = QRadioButton("Load Points")
        #select_pointwise = QRadioButton("Point-wise")
        select_load_points.setChecked(True)  # Default selection
        panel2_row2.addWidget(select_grid)
        panel2_row2.addWidget(select_load_points)
        #panel2_row2.addWidget(select_pointwise)
        panel2_layout.addLayout(panel2_row2)

        self.button_group_ptype = QButtonGroup()
        self.button_group_ptype.addButton(select_grid, 0)  # ID = 0
        self.button_group_ptype.addButton(select_load_points, 1)  # ID = 1
        #self.button_group_ptype.addButton(select_pointwise, 2)  # ID = 2

        self.pointselection = QStackedWidget()

        grid_panel = QGroupBox("Grid options")

        grid_layout1 = QGridLayout()
        grid_panel.setLayout(grid_layout1)
        grid_layout1.addWidget(QLabel('min. latitude'), 0, 0)
        self.latmin = QLineEdit()
        self.latmin.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.latmin, 1, 0)
        grid_layout1.addWidget(QLabel('latitude step'), 0, 1)
        self.latstep = QLineEdit()
        self.latstep.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.latstep, 1, 1)
        grid_layout1.addWidget(QLabel('max. latitude'), 0, 2)
        self.latmax = QLineEdit()
        self.latmax.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.latmax, 1, 2)
        grid_layout1.addWidget(QLabel('min. longitude'), 2, 0)
        self.lonmin = QLineEdit()
        self.lonmin.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.lonmin, 3, 0)
        grid_layout1.addWidget(QLabel('longitude step'), 2, 1)
        self.lonstep = QLineEdit()
        self.lonstep.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.lonstep, 3, 1)
        grid_layout1.addWidget(QLabel('max. longitude'), 2, 2)
        self.lonmax = QLineEdit()
        self.lonmax.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.lonmax, 3, 2)
        ########
        unit_list = ["degrees","minutes","seconds"]
        grid_layout1.addWidget(QLabel("Unit of grid resolution"), 2, 3)
        self.unit_combo_box = QComboBox()
        self.unit_combo_box.addItems(unit_list)
        self.unit_combo_box.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.unit_combo_box, 3, 3)
        grid_layout1.addWidget(QLabel('Height above reference surface [m]'),0,3)
        self.height_above_surface = QLineEdit()
        self.height_above_surface.setText('0.0')
        self.height_above_surface.setMaximumWidth(int(width/7))
        grid_layout1.addWidget(self.height_above_surface,1,3)


        load_panel = QGroupBox("Load points options")
        #load_panel.setLayout(QVBoxLayout())
        ## get points
        load_point_layout = QVBoxLayout()
        load_point_layout1 = QHBoxLayout()
        browse_input_points = QPushButton("... Browse")
        browse_input_points.setStyleSheet("QPushButton { background-color: rgb(200, 200, 255); }")
        browse_input_points.clicked.connect(self.open_points_dialog)
        self.input_points_file = QLineEdit()
        self.input_points_file.setReadOnly(True)
        load_point_layout1.addWidget(browse_input_points)
        load_point_layout1.addWidget(self.input_points_file)

        load_point_layout2 = QHBoxLayout()
        self.use_point_numbers = QCheckBox("Point numbers used in input points file.")
        load_point_layout2.addWidget(self.use_point_numbers)

        load_point_layout.addLayout(load_point_layout1)
        load_point_layout.addLayout(load_point_layout2)
        ##
        load_panel.setLayout(load_point_layout)

        #pointwise_panel = QGroupBox("Pointwise options")
        #pointwise_panel.setLayout(QVBoxLayout())
        #pointwise_panel.layout().addWidget(QLabel("Configure pointwise parameters..."))

        self.pointselection.addWidget(grid_panel)      # index 0
        self.pointselection.addWidget(load_panel)      # index 1
        #self.pointselection.addWidget(pointwise_panel) # index 2

        self.button_group_ptype.buttonClicked[int].connect(self.pointselection.setCurrentIndex)

        # default
        self.pointselection.setCurrentIndex(1)

        panel2_layout.addWidget(self.pointselection)

        # Panel 3: Calculated parameters and output selection (empty for now)
        panel3 = QGroupBox("Calculated parameters and output selection")
        panel3.setStyleSheet("QGroupBox { background-color: rgb(235, 220, 220); }")
        panel_3_layout = QHBoxLayout()
        panel3.setLayout(panel_3_layout)  # Empty panel for now

        # Add widgets to panel 3 layout
        self.quantity_list =["","Gravitational potential",'topography','Disturbing potential','Gravity potential','Gravity anomaly', 'Gravity disturbance' ,'Gravity vector', 'Gravity',\
                        "Deflection of vertical - xi", "Deflection of vertical - eta", "Deflection of vertical - theta", "Geoid undulation", "Height anomaly", "Pseudo height anomaly",\
                        'V_xz','V_yz' ,'V_xy', 'V_xx','V_yy','V_zz', 'V_delta', 'W_xz' ,'W_yz' ,'W_xy', 'W_xx' , 'W_yy', 'W_zz', 'W_delta'  , \
                        'T_xz' ,'T_yz' ,'T_xy', 'T_xx' , 'T_yy', 'T_zz' , 'T_delta']
        #quantity_list = ["","Gravitational potential", "Disturbing potential", "Gravity anomaly"]
        combo_selection = QVBoxLayout()

        self.quantity_combo_box1 = QComboBox()
        self.quantity_combo_box1.addItems(self.quantity_list)
        combo_selection.addWidget(self.quantity_combo_box1)

        self.quantity_combo_box2 = QComboBox()
        self.quantity_combo_box2.addItems(self.quantity_list)
        combo_selection.addWidget(self.quantity_combo_box2)

        self.quantity_combo_box3 = QComboBox()
        self.quantity_combo_box3.addItems(self.quantity_list)
        combo_selection.addWidget(self.quantity_combo_box3)

        # Enable/disable combo boxes 2 and 3 based on point type selection
        self.button_group_ptype.buttonClicked[int].connect(self.update_combo_boxes_state)
        
        self.outfile_name = QLineEdit()
        set_output_file = QPushButton("Set output file")
        self.outfile_name.setReadOnly(True)
        self.report = QCheckBox('Generate report')

        #panel_3_layout.addWidget(self.quantity_combo_box)
        panel_3_layout.addLayout(combo_selection)
        panel_3_layout.addWidget(set_output_file)
        panel_3_layout.addWidget(self.outfile_name)
        panel_3_layout.addWidget(self.report)
        set_output_file.clicked.connect(self.open_file_dialog_outfile)

        # Panel 4: Buttons for run and exit
        panel4 = QGroupBox()
        panel4_layout = QVBoxLayout()

        panel4_layout1 = QHBoxLayout()
        StartButton = QPushButton('OK')

        StartButton.clicked.connect(self.process_function)

        panel4_layout1.addWidget(StartButton)
        CloseButton = QPushButton('Close')
        CloseButton.clicked.connect(self.close)
        CloseButton.clicked.connect(QCoreApplication.instance().quit)
        panel4_layout1.addWidget(CloseButton)

        panel4_layout2 = QHBoxLayout()
        self.feedback = QLabel('')
        panel4_layout2.addWidget(self.feedback)

        panel4_layout.addLayout(panel4_layout1)
        panel4_layout.addLayout(panel4_layout2)

        panel4.setLayout(panel4_layout)


        # Add panels to main layout
        main_layout.addWidget(panel1)
        main_layout.addWidget(panel2)
        main_layout.addWidget(panel3)
        main_layout.addWidget(panel4)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select GGM File", "", "All Files (*)")
        if file_name:
            self.imported_ggm_file.setText(file_name)
    def open_file_dialog_topo(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select file containing SH coefficients of topography", "", "All Files (*)")
        if file_name:
            self.imported_SH_topo_file.setText(file_name)
    def open_points_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select point data file", "", "All Files (*)")
        if file_name:
            self.input_points_file.setText(file_name)
    def open_file_dialog_outfile(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Set Output File", "", "All Files (*)")
        if file_name:
            self.outfile_name.setText(file_name)
    def toggle_nmin_input(self, checked):
        if checked:
            self.nmax_input.clear()
            self.nmax_input.setReadOnly(True)
            self.nmax_input.setStyleSheet("QLineEdit { background-color: lightgray; }")
            #print(self.imported_ggm_file.text())
            #print(self.button_group_sh_ell.checkedButton().text())
        else:
            self.nmax_input.setReadOnly(False)
            self.nmax_input.setStyleSheet("QLineEdit { background-color: white; }")
    
    def update_combo_boxes_state(self, point_type_id):
            if point_type_id == 0:  # Grid
                self.quantity_combo_box2.setEnabled(False)
                self.quantity_combo_box2.setCurrentIndex(0)
                self.quantity_combo_box3.setEnabled(False)
                self.quantity_combo_box2.setCurrentIndex(0)
            elif point_type_id == 1:  # Load points
                self.quantity_combo_box2.setEnabled(True)
                self.quantity_combo_box3.setEnabled(True)
            #elif point_type_id == 2:  # Pointwise
            #    self.quantity_combo_box2.setEnabled(True)
            #    self.quantity_combo_box3.setEnabled(True)
    def toggle_GM_R(self, checked):
        if checked:
            self.GM_input.clear()
            self.GM_input.setReadOnly(True)
            self.GM_input.setStyleSheet("QLineEdit { background-color: lightgray; }")

            self.scale_input.clear()
            self.scale_input.setReadOnly(True)
            self.scale_input.setStyleSheet("QLineEdit { background-color: lightgray; }")
        else:
            self.GM_input.setReadOnly(False)
            self.GM_input.setStyleSheet("QLineEdit { background-color: white; }")

            self.scale_input.setReadOnly(False)
            self.scale_input.setStyleSheet("QLineEdit { background-color: white; }")

    def process_function(self):
        self.feedback.setText('Computing...')
        QCoreApplication.processEvents()
        shcs_data = self.imported_ggm_file.text()
        if len(shcs_data) == 0:
            self.feedback.setText('ERROR! Geoptotential model not chosen.')
            QCoreApplication.processEvents()
            return
        use_max_GGM = self.use_max_GGM.isChecked()
        try:
            nmin = int(self.nmin_input.text())
            nmax = None if use_max_GGM else int(self.nmax_input.text())
            GM = None if self.use_GM_R.isChecked() else float(self.GM_input.text())
            R = None if self.use_GM_R.isChecked() else float(self.scale_input.text())
        except:
            self.feedback.setText('ERROR! Geoptotential model parameters not set correctly.')
            QCoreApplication.processEvents()
            return
        normal_field_removed = self.normal_field_removed.isChecked()
        ellipsoid = self.ell_combo_box.currentText()
        print(ellipsoid)
        
        DTM_shcs_data = self.imported_SH_topo_file.text()
        if len(DTM_shcs_data) == 0:
            DTM_shcs_data = None


        # quantity
        q1_index = self.quantity_combo_box1.currentIndex()
        q2_index = self.quantity_combo_box2.currentIndex()
        q3_index = self.quantity_combo_box3.currentIndex()
        quantity_list_short = [None, 'V', 'topo', 'T', 'W', 'dg', 'dg_dist', 'g', 'g_abs', 'xi', 'eta', 'theta','N', 'zeta', \
                               'zeta_ell','V_xz', 'V_yz', 'V_xy', 'V_xx', 'V_yy', 'V_zz', 'V_delta', 'W_xz', 'W_yz', 'W_xy', \
                                'W_xx', 'W_yy', 'W_zz', 'W_delta', 'T_xz', 'T_yz', 'T_xy', 'T_xx', 'T_yy', 'T_zz', 'T_delta']
        quantity = [quantity_list_short[q1_index],quantity_list_short[q2_index],quantity_list_short[q3_index]]
        quantity_long = [self.quantity_list[q1_index],self.quantity_list[q2_index],self.quantity_list[q3_index]]
        quantity = [item for item in quantity if item is not None]
        quantity_long = [item for item in quantity_long if len(item)>0]
        if len(list(set(quantity))) != len(quantity):
            self.feedback.setText('ERROR! Duplicates in selected quantities.')
            QCoreApplication.processEvents()
            return
        if len(quantity) == 0:
            self.feedback.setText('ERROR! No quantity selected for synthesis.')
            QCoreApplication.processEvents()
            return
        if len(quantity) == 1:
            quantity = quantity[0]
        
        output_file = self.outfile_name.text()
        if len(output_file) == 0:
            self.feedback.setText('ERROR! Output file not specified.')
            QCoreApplication.processEvents()
            return
        
        if self.report.isChecked():
            report_file = os.path.splitext(output_file)[0] + 'report.txt'
            report = open(report_file,'w')
            
            print(f'Output file:                                {output_file}',file=report)
            print(f'Geopotential model file:                    {shcs_data}',file=report)
            print(f'GM of the geopotential model (m^3*s^-2):    {GM}',file=report)
            print(f'R of the geopotential model (m):            {R}',file=report)
            print(f'Minimum used degree:                        {nmin}',file=report)
            print(f'Maximum used degree:                        {nmax}',file=report)
            print(f'Reference ellipsoid:                        {ellipsoid}',file=report)
            print(f'Computed:                                   {quantity_long}',file=report)

        if self.button_group_ptype.checkedId() == 0:  # Grid
            try:
                min_lat = float(self.latmin.text())
                _latstep = float(self.latstep.text())
                max_lat = float(self.latmax.text())
                min_lon = float(self.lonmin.text())
                _lonstep = float(self.lonstep.text())
                max_lon = float(self.lonmax.text())
                resolution = (_latstep, _lonstep)
                height = float(self.height_above_surface.text())
                ref_surface_type = 'ellipsoid' if self.button_group_sh_ell.checkedId() == 1 else 'sphere'
                resolution_unit = self.unit_combo_box.currentText()
                print(f"Grid selected with latmin={min_lat}, latmax={max_lat}, lonmin={min_lon}, resolution={resolution}, lonmax={max_lon}, unit={resolution_unit}, height_above_surface={height}")
                #print(f"Reference surface type: {ref_surface_type}")
            except:
                self.feedback.setText('ERROR! Some mandatory input parameters not set.')
                QCoreApplication.processEvents()
                return
            if self.report.isChecked():
                t1 = datetime.datetime.now()
                print(f'Computation started:                        {t1}',file=report)
                print(f'Type of reference surface:                  {ref_surface_type}',file=report)
                print(f'Latitude limit North (deg):                 {max_lat}',file=report)
                print(f'Latitude limit South (deg):                 {min_lat}',file=report)
                print(f'Longitude limit West (deg):                 {min_lon}',file=report)
                print(f'Longitude limit East (deg):                 {max_lon}',file=report)
                print(f'Height above reference surface (m):         {height}',file=report)
            try:
                n_lat,n_lon,n_points=calc_grid(output_file,quantity , min_lat , max_lat , min_lon , max_lon , resolution , shcs_data , resolution_unit , nmin , nmax , ellipsoid ,ref_surface_type , height ,GM , R , DTM_shcs_data , normal_field_removed)
            except ValueError:
                self.feedback.setText('ERROR! Not recognised output file type.')
                QCoreApplication.processEvents()
                return
            if self.report.isChecked():
                print(f'Grid points in latitude direction:          {n_lat}',file=report)
                print(f'Grid points in longitude direction:         {n_lon}',file=report)
                print(f'Number of grid points:                      {n_points}',file=report)
                t2 = datetime.datetime.now()
                dt = t2-t1
                print(f'Computation time:                           {dt}',file=report)
                report.close()
                
        elif self.button_group_ptype.checkedId() == 1:  # Load points
            input_file = self.input_points_file.text()
            point_numbers = self.use_point_numbers.isChecked()
            points_type = 'ellipsoidal' if self.button_group_sh_ell.checkedId() == 1 else 'spherical'

            if len(input_file) == 0:
                self.feedback.setText('ERROR! NO INPUT FILE GIVEN.')
                QCoreApplication.processEvents()
                return

            if self.report.isChecked():
                print(f'Input file:                                 {input_file}',file=report)
                t1 = datetime.datetime.now()
                print(f'Computation started:                        {t1}',file=report)
                print(f'Type of the input coordinates:              {points_type}',file=report)

            n_points=calc_point(input_file, point_numbers,output_file ,shcs_data, points_type, quantity , nmin , nmax , ellipsoid , GM , R , DTM_shcs_data , normal_field_removed)
            if self.report.isChecked():
                print(f'Number of points:                           {n_points}',file=report)
                t2 = datetime.datetime.now()
                dt = t2-t1
                print(f'Computation time:                           {dt}',file=report)
                if points_type == 'ellipsoidal':
                    if point_numbers:
                        columns = ['Point number', 'Ellipsoidal latitude (deg)', 'Longitude (deg)', 'Ellipsoidal height (m)']
                    else:
                        columns = ['Ellipsoidal latitude (deg)', 'Longitude (deg)', 'Ellipsoidal height (m)']
                else:
                    if point_numbers:
                        columns = ['Point number', 'Spherical latitude (deg)', 'Longitude (deg)', 'Spherical radius (m)']
                    else:
                        columns = ['Spherical latitude (deg)', 'Longitude (deg)', 'Spherical radius (m)']
                columns.extend(quantity_long)
                columns_str = ' | '.join(columns)
                print('Exported data file contains the following columns:',file=report)
                print(columns_str,file=report)
                report.close()
  
                
            QCoreApplication.processEvents()
        #else:
        #    pass

        self.feedback.setText('Computation finished.')

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
if __name__ == "__main__":
    main()
