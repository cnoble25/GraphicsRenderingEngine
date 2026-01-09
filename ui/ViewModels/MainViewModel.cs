using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using GraphicsRendererUI.Models;
using GraphicsRendererUI.Utils;
using Avalonia.Controls;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Avalonia.Platform;
using Avalonia;
using Models = GraphicsRendererUI.Models;

namespace GraphicsRendererUI.ViewModels
{
    public class MainViewModel : ObservableObject
    {
        private IntPtr _sceneHandle = IntPtr.Zero;
        private string _statusMessage = "Ready";
        private string _renderWidth = "800";
        private string _renderHeight = "450";
        private double _luminosity = 5.0;
        private string _outputPath = "";
        private int _maxBounces = 3; // Default to 3 bounces
        private int _compressionLevel = 1; // Default to 1 (no compression)
        private Bitmap? _renderedImage;
        private int _focusPointX = 0;
        private int _focusPointY = 0;
        public ObservableCollection<SceneObject> Objects { get; } = new ObservableCollection<SceneObject>();
        public ObservableCollection<Light> Lights { get; } = new ObservableCollection<Light>();
        
        // Unified collection combining objects and lights
        public ObservableCollection<ISceneItem> SceneItems { get; } = new ObservableCollection<ISceneItem>();

        private ISceneItem? _selectedItem;
        public ISceneItem? SelectedItem
        {
            get => _selectedItem;
            set
            {
                SetProperty(ref _selectedItem, value);
                OnPropertyChanged(nameof(HasSelectedItem));
                OnPropertyChanged(nameof(SelectedObject));
                OnPropertyChanged(nameof(SelectedLight));
                OnPropertyChanged(nameof(HasSelectedObject));
                OnPropertyChanged(nameof(HasSelectedLight));
            }
        }

        public bool HasSelectedItem => SelectedItem != null;
        
        // Convenience properties for backward compatibility
        public SceneObject? SelectedObject
        {
            get => SelectedItem as SceneObject;
            set => SelectedItem = value;
        }

        public bool HasSelectedObject => SelectedItem is SceneObject;
        
        public Light? SelectedLight
        {
            get => SelectedItem as Light;
            set => SelectedItem = value;
        }

        public bool HasSelectedLight => SelectedItem is Light;

        public string StatusMessage
        {
            get => _statusMessage;
            set => SetProperty(ref _statusMessage, value);
        }

        public string RenderWidth
        {
            get => _renderWidth;
            set
            {
                SetProperty(ref _renderWidth, value);
                // Clamp focus point when width changes
                if (int.TryParse(value, out int width) && width > 0)
                {
                    FocusPointX = Math.Min(FocusPointX, width - 1);
                }
            }
        }

        public string RenderHeight
        {
            get => _renderHeight;
            set
            {
                SetProperty(ref _renderHeight, value);
                // Clamp focus point when height changes
                if (int.TryParse(value, out int height) && height > 0)
                {
                    FocusPointY = Math.Min(FocusPointY, height - 1);
                }
            }
        }

        public double Luminosity
        {
            get => _luminosity;
            set => SetProperty(ref _luminosity, value);
        }

        public string OutputPath
        {
            get => _outputPath;
            set => SetProperty(ref _outputPath, value);
        }

        public int MaxBounces
        {
            get => _maxBounces;
            set => SetProperty(ref _maxBounces, Math.Max(0, Math.Min(10, value))); // Clamp to [0, 10]
        }

        public int CompressionLevel
        {
            get => _compressionLevel;
            set => SetProperty(ref _compressionLevel, value); // Don't clamp immediately - allow typing
        }

        // Validate and clamp compression level (called when text box loses focus)
        public void ValidateCompressionLevel()
        {
            int clamped = Math.Max(1, Math.Min(32, _compressionLevel));
            if (clamped != _compressionLevel)
            {
                CompressionLevel = clamped;
                StatusMessage = $"Compression level adjusted to {clamped} (valid range: 1-32)";
            }
        }

        public Bitmap? RenderedImage
        {
            get => _renderedImage;
            set => SetProperty(ref _renderedImage, value);
        }

        public int FocusPointX
        {
            get => _focusPointX;
            set
            {
                // Clamp to current render width (or 0 if not set)
                int maxX = 0;
                if (int.TryParse(RenderWidth, out int width) && width > 0)
                    maxX = width - 1;
                int clampedValue = Math.Max(0, Math.Min(maxX, value));
                SetProperty(ref _focusPointX, clampedValue);
            }
        }

        public int FocusPointY
        {
            get => _focusPointY;
            set
            {
                // Clamp to current render height (or 0 if not set)
                int maxY = 0;
                if (int.TryParse(RenderHeight, out int height) && height > 0)
                    maxY = height - 1;
                int clampedValue = Math.Max(0, Math.Min(maxY, value));
                SetProperty(ref _focusPointY, clampedValue);
            }
        }

        public ICommand AddPyramidCommand { get; }
        public ICommand AddBoxCommand { get; }
        public ICommand AddPlaneCommand { get; }
        public ICommand AddObjFileCommand { get; }
        public ICommand RemoveObjectCommand { get; }
        public ICommand AddLightCommand { get; }
        public ICommand RemoveLightCommand { get; }
        public ICommand RenderCommand { get; }
        public ICommand BrowseOutputPathCommand { get; }
        public ICommand ClearFocusPointCommand { get; }

        public MainViewModel()
        {
            _sceneHandle = RendererAPI.create_scene();
            if (_sceneHandle == IntPtr.Zero)
            {
                StatusMessage = "Failed to create scene!";
            }

            AddPyramidCommand = new RelayCommand(AddPyramid);
            AddBoxCommand = new RelayCommand(AddBox);
            AddPlaneCommand = new RelayCommand(AddPlane);
            AddObjFileCommand = new RelayCommand(AddObjFile);
            RemoveObjectCommand = new RelayCommand(RemoveSelectedObject, () => HasSelectedObject);
            AddLightCommand = new RelayCommand(AddLight);
            RemoveLightCommand = new RelayCommand(RemoveSelectedLight, () => HasSelectedLight);
            
            // Update command state when selection changes
            PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(SelectedItem) || e.PropertyName == nameof(SelectedObject))
                {
                    (RemoveObjectCommand as RelayCommand)?.RaiseCanExecuteChanged();
                }
                if (e.PropertyName == nameof(SelectedItem) || e.PropertyName == nameof(SelectedLight))
                {
                    (RemoveLightCommand as RelayCommand)?.RaiseCanExecuteChanged();
                }
            };
            RenderCommand = new RelayCommand(RenderScene);
            BrowseOutputPathCommand = new RelayCommand(BrowseOutputPath);
            ClearFocusPointCommand = new RelayCommand(ClearFocusPoint);
        }

        private void ClearFocusPoint()
        {
            FocusPointX = 0;
            FocusPointY = 0;
            StatusMessage = "Focus point cleared";
        }

        private void AddPyramid()
        {
            var obj = new SceneObject
            {
                Name = $"Pyramid {GetNextObjectNumber("Pyramid")}",
                ObjectType = ObjectType_API.OBJECT_TYPE_PYRAMID
            };
            AddObject(obj);
        }

        private void AddBox()
        {
            var obj = new SceneObject
            {
                Name = $"Box {GetNextObjectNumber("Box")}",
                ObjectType = ObjectType_API.OBJECT_TYPE_BOX
            };
            AddObject(obj);
        }

        private void AddPlane()
        {
            var obj = new SceneObject
            {
                Name = $"Plane {GetNextObjectNumber("Plane")}",
                ObjectType = ObjectType_API.OBJECT_TYPE_PLANE
            };
            AddObject(obj);
        }

        private int GetNextObjectNumber(string prefix)
        {
            int maxNumber = 0;
            foreach (var obj in Objects)
            {
                if (obj.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    var parts = obj.Name.Split(' ');
                    if (parts.Length > 1 && int.TryParse(parts[parts.Length - 1], out int number))
                    {
                        maxNumber = Math.Max(maxNumber, number);
                    }
                }
            }
            return maxNumber + 1;
        }

        private int GetNextLightNumber()
        {
            int maxNumber = 0;
            foreach (var light in Lights)
            {
                if (light.Name.StartsWith("Light", StringComparison.OrdinalIgnoreCase))
                {
                    var parts = light.Name.Split(' ');
                    if (parts.Length > 1 && int.TryParse(parts[parts.Length - 1], out int number))
                    {
                        maxNumber = Math.Max(maxNumber, number);
                    }
                }
            }
            return maxNumber + 1;
        }

        private async void AddObjFile()
        {
            try
            {
                var app = Avalonia.Application.Current;
                if (app?.ApplicationLifetime is Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime desktop)
                {
                    var window = desktop.MainWindow;
                    if (window != null)
                    {
                        var topLevel = TopLevel.GetTopLevel(window);
                        if (topLevel != null)
                        {
                            var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
                            {
                                Title = "Select OBJ File",
                                FileTypeFilter = new[] { 
                                    FilePickerFileTypes.All,
                                    new FilePickerFileType("OBJ Files") { Patterns = new[] { "*.obj" } }
                                }
                            });

                            if (files.Count >= 1)
                            {
                                var file = files[0];
                                var path = file.Path.LocalPath;
                                var obj = new SceneObject
                                {
                                    Name = Path.GetFileName(path),
                                    ObjectType = ObjectType_API.OBJECT_TYPE_OBJ_FILE,
                                    ObjFilePath = path
                                };
                                AddObject(obj);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error opening file: {ex.Message}";
            }
        }

        private void AddObject(SceneObject obj)
        {
            // Add to native scene first
            var api = obj.ToAPI();
            
            int result = RendererAPI.add_object_to_scene(_sceneHandle, ref api);

            if (result == 0)
            {
                StatusMessage = $"Failed to add {obj.Name} to scene";
                return;
            }

            // Only add to UI collection if native add succeeded
            Objects.Add(obj);
            SceneItems.Add(obj);
            SelectedItem = obj;
            StatusMessage = $"Added {obj.Name} to scene";
        }

        private void RemoveSelectedObject()
        {
            if (SelectedObject == null) return;

            int index = Objects.IndexOf(SelectedObject);
            if (index >= 0)
            {
                int result = RendererAPI.remove_object_from_scene(_sceneHandle, index);
                if (result != 0)
                {
                    var obj = Objects[index];
                    Objects.RemoveAt(index);
                    SceneItems.Remove(obj);
                    SelectedItem = SceneItems.FirstOrDefault();
                    StatusMessage = "Object removed";
                }
                else
                {
                    StatusMessage = "Failed to remove object";
                }
            }
        }

        private void AddLight()
        {
            var light = new Light
            {
                Name = $"Light {GetNextLightNumber()}",
                PositionX = 0.0,
                PositionY = 5.0,
                PositionZ = 5.0,
                ColorR = 1.0,
                ColorG = 1.0,
                ColorB = 1.0,
                Luminosity = 5.0
            };

            var lightAPI = new Light_API
            {
                position = new Vec3_API { x = light.PositionX, y = light.PositionY, z = light.PositionZ },
                color = new Vec3_API { x = light.ColorR, y = light.ColorG, z = light.ColorB },
                luminosity = light.Luminosity
            };

            int result = RendererAPI.add_light_to_scene(_sceneHandle, ref lightAPI);
            if (result != 0)
            {
                light.Index = Lights.Count;
                Lights.Add(light);
                SceneItems.Add(light);
                SelectedItem = light;
                StatusMessage = $"Added {light.Name} to scene";
            }
            else
            {
                StatusMessage = "Failed to add light to scene";
            }
        }

        private void RemoveSelectedLight()
        {
            if (SelectedLight == null) return;

            int index = Lights.IndexOf(SelectedLight);
            if (index >= 0)
            {
                int result = RendererAPI.remove_light_from_scene(_sceneHandle, index);
                if (result != 0)
                {
                    var light = Lights[index];
                    Lights.RemoveAt(index);
                    SceneItems.Remove(light);
                    // Update indices
                    for (int i = 0; i < Lights.Count; i++)
                    {
                        Lights[i].Index = i;
                    }
                    SelectedItem = SceneItems.FirstOrDefault();
                    StatusMessage = "Light removed";
                }
                else
                {
                    StatusMessage = "Failed to remove light";
                }
            }
        }

        private void RenderScene()
        {
            if (_sceneHandle == IntPtr.Zero)
            {
                StatusMessage = "Scene not initialized!";
                return;
            }

            // Validate compression level before rendering (in case user didn't tab off)
            ValidateCompressionLevel();

            // Update all object transforms and reflectivities
            for (int i = 0; i < Objects.Count; i++)
            {
                var obj = Objects[i];
                // Default scale to 1.0 if it's 0 (allows placeholder to show)
                var scale = obj.Scale;
                if (scale.x == 0 && scale.y == 0 && scale.z == 0)
                {
                    scale = new Vec3_API { x = 1.0, y = 1.0, z = 1.0 };
                }
                
                // Convert rotation to radians if currently in degrees (API expects radians)
                Rotation_API rotation = obj.Rotation;
                if (obj.UseDegrees)
                {
                    rotation = new Rotation_API
                    {
                        roll = obj.Rotation.roll * Math.PI / 180.0,
                        pitch = obj.Rotation.pitch * Math.PI / 180.0,
                        yaw = obj.Rotation.yaw * Math.PI / 180.0
                    };
                }
                
                var transform = new Transform_API
                {
                    position = obj.Position,
                    scale = scale,
                    rotation = rotation
                };
                RendererAPI.update_object_transform(_sceneHandle, i, ref transform);
                
                // Update light absorption
                RendererAPI.update_object_light_absorption(_sceneHandle, i, obj.LightAbsorption);
            }

            // Update all lights
            for (int i = 0; i < Lights.Count; i++)
            {
                var light = Lights[i];
                var lightAPI = new Light_API
                {
                    position = new Vec3_API { x = light.PositionX, y = light.PositionY, z = light.PositionZ },
                    color = new Vec3_API { x = light.ColorR, y = light.ColorG, z = light.ColorB },
                    luminosity = light.Luminosity
                };
                RendererAPI.update_light(_sceneHandle, i, ref lightAPI);
            }

            StatusMessage = "Rendering...";
            RenderedImage = null; // Clear previous image
            
            // Parse render dimensions with defaults
            if (!int.TryParse(RenderWidth, out int width) || width <= 0)
                width = 800;
            if (!int.TryParse(RenderHeight, out int height) || height <= 0)
                height = 450;
            
            try
            {
                // Render directly to buffer for fast display (no file I/O, pixel buffer method)
                int bufferSize = width * height * 4; // RGBA, 4 bytes per pixel
                byte[] buffer = new byte[bufferSize];
                
                // Clamp focus point to resolution before rendering
                int clampedFocusX = Math.Max(0, Math.Min(width - 1, FocusPointX));
                int clampedFocusY = Math.Max(0, Math.Min(height - 1, FocusPointY));
                
                int result = RendererAPI.render_scene_to_buffer(_sceneHandle, buffer, width, height, MaxBounces, clampedFocusX, clampedFocusY, CompressionLevel);
                
                // SUCCESS = 1 (RendererError::SUCCESS)
                if (result == 1)
                {
                    // Create Avalonia bitmap directly from buffer
                    Dispatcher.UIThread.Post(() =>
                    {
                        try
                        {
                            // Store old image reference before setting new one
                            var oldImage = _renderedImage;
                            
                            // Create WriteableBitmap from buffer
                            var writeableBitmap = new WriteableBitmap(
                                new PixelSize(width, height),
                                new Vector(96, 96),
                                PixelFormat.Bgra8888,
                                AlphaFormat.Opaque);
                            
                            using (var lockedBitmap = writeableBitmap.Lock())
                            {
                                System.Runtime.InteropServices.Marshal.Copy(buffer, 0, lockedBitmap.Address, bufferSize);
                            }
                            
                            // Set the new image
                            RenderedImage = writeableBitmap;
                            
                            // Dispose old image after a short delay
                            Dispatcher.UIThread.Post(() =>
                            {
                                if (oldImage != null)
                                {
                                    oldImage.Dispose();
                                }
                            }, DispatcherPriority.Background);
                            
                            StatusMessage = $"Rendered ({width}x{height}, comp={CompressionLevel}) - Check console for timing and saved image path";
                        }
                        catch (Exception ex)
                        {
                            StatusMessage = $"Failed to create image: {ex.Message}";
                        }
                    }, DispatcherPriority.Render);
                }
                else
                {
                    // Pixel buffer rendering failed - show error
                    StatusMessage = $"Pixel buffer render failed! Error code: {result}. Check console for details.";
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Render error: {ex.Message}";
            }
        }

        private async void BrowseOutputPath()
        {
            try
            {
                var app = Avalonia.Application.Current;
                if (app?.ApplicationLifetime is Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime desktop)
                {
                    var window = desktop.MainWindow;
                    if (window != null)
                    {
                        var topLevel = TopLevel.GetTopLevel(window);
                        if (topLevel != null)
                        {
                            var file = await topLevel.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
                            {
                                Title = "Save Output Image",
                                SuggestedFileName = OutputPath,
                                FileTypeChoices = new[] { 
                                    FilePickerFileTypes.All,
                                    new FilePickerFileType("PPM Files") { Patterns = new[] { "*.ppm" } },
                                    new FilePickerFileType("JPG Files") { Patterns = new[] { "*.jpg", "*.jpeg" } }
                                }
                            });

                            if (file != null)
                            {
                                OutputPath = file.Path.LocalPath;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error saving file: {ex.Message}";
            }
        }

        ~MainViewModel()
        {
            if (_sceneHandle != IntPtr.Zero)
            {
                RendererAPI.free_scene(_sceneHandle);
            }
        }
    }
}
