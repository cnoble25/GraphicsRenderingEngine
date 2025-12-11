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
using Models = GraphicsRendererUI.Models;

namespace GraphicsRendererUI.ViewModels
{
    public class MainViewModel : ObservableObject
    {
        private IntPtr _sceneHandle = IntPtr.Zero;
        private string _statusMessage = "Ready";
        private int _renderWidth = 800;
        private int _renderHeight = 450;
        private double _luminosity = 5.0;
        private string _outputPath = "renders/output.ppm";
        private Bitmap? _renderedImage;
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

        public int RenderWidth
        {
            get => _renderWidth;
            set => SetProperty(ref _renderWidth, value);
        }

        public int RenderHeight
        {
            get => _renderHeight;
            set => SetProperty(ref _renderHeight, value);
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

        public Bitmap? RenderedImage
        {
            get => _renderedImage;
            set => SetProperty(ref _renderedImage, value);
        }

        public ICommand AddPyramidCommand { get; }
        public ICommand AddBoxCommand { get; }
        public ICommand AddObjFileCommand { get; }
        public ICommand RemoveObjectCommand { get; }
        public ICommand AddLightCommand { get; }
        public ICommand RemoveLightCommand { get; }
        public ICommand RenderCommand { get; }
        public ICommand BrowseOutputPathCommand { get; }

        public MainViewModel()
        {
            _sceneHandle = RendererAPI.create_scene();
            if (_sceneHandle == IntPtr.Zero)
            {
                StatusMessage = "Failed to create scene!";
            }

            AddPyramidCommand = new RelayCommand(AddPyramid);
            AddBoxCommand = new RelayCommand(AddBox);
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
        }

        private void AddPyramid()
        {
            var obj = new SceneObject
            {
                Name = $"Pyramid {Objects.Count + 1}",
                ObjectType = ObjectType_API.OBJECT_TYPE_PYRAMID
            };
            AddObject(obj);
        }

        private void AddBox()
        {
            var obj = new SceneObject
            {
                Name = $"Box {Objects.Count + 1}",
                ObjectType = ObjectType_API.OBJECT_TYPE_BOX
            };
            AddObject(obj);
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
                StatusMessage = $"Added light {Lights.Count} to scene";
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

            // Update all object transforms
            for (int i = 0; i < Objects.Count; i++)
            {
                var obj = Objects[i];
                var transform = new Transform_API
                {
                    position = obj.Position,
                    scale = obj.Scale,
                    rotation = obj.Rotation
                };
                RendererAPI.update_object_transform(_sceneHandle, i, ref transform);
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
            
            try
            {
                int result = RendererAPI.render_scene(_sceneHandle, OutputPath, RenderWidth, RenderHeight, Luminosity, RenderMode_API.RENDER_MODE_RAY_TRACING);
                if (result != 0)
                {
                    StatusMessage = $"Rendered to {OutputPath}";
                    
                    // Load and display the rendered image
                    // Note: renderer_api.cpp automatically changes extension to .ppm
                    try
                    {
                        string ppmPath = OutputPath;
                        if (!ppmPath.EndsWith(".ppm", StringComparison.OrdinalIgnoreCase))
                        {
                            int dotPos = ppmPath.LastIndexOf('.');
                            if (dotPos >= 0)
                            {
                                ppmPath = ppmPath.Substring(0, dotPos) + ".ppm";
                            }
                            else
                            {
                                ppmPath += ".ppm";
                            }
                        }
                        
                        if (File.Exists(ppmPath))
                        {
                            RenderedImage = PpmDecoder.DecodePpm(ppmPath);
                            StatusMessage = $"Rendered to {ppmPath} - Image displayed";
                        }
                        else
                        {
                            StatusMessage = $"Rendered to {ppmPath} - File not found";
                        }
                    }
                    catch (Exception ex)
                    {
                        StatusMessage = $"Rendered to {OutputPath} - Failed to load image: {ex.Message}";
                    }
                }
                else
                {
                    StatusMessage = "Render failed!";
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
