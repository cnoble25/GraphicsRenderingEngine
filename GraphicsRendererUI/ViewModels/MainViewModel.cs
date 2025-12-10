using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using GraphicsRendererUI.Models;
using Avalonia.Controls;
using Avalonia.Platform.Storage;

namespace GraphicsRendererUI.ViewModels
{
    public class MainViewModel : ObservableObject
    {
        private IntPtr _sceneHandle = IntPtr.Zero;
        private SceneObject? _selectedObject;
        private string _statusMessage = "Ready";
        private int _renderWidth = 800;
        private int _renderHeight = 450;
        private double _luminosity = 5.0;
        private string _outputPath = "output.ppm";

        public ObservableCollection<SceneObject> Objects { get; } = new ObservableCollection<SceneObject>();

        public SceneObject? SelectedObject
        {
            get => _selectedObject;
            set
            {
                SetProperty(ref _selectedObject, value);
                OnPropertyChanged(nameof(HasSelectedObject));
            }
        }

        public bool HasSelectedObject => SelectedObject != null;

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

        public ICommand AddPyramidCommand { get; }
        public ICommand AddBoxCommand { get; }
        public ICommand AddObjFileCommand { get; }
        public ICommand RemoveObjectCommand { get; }
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
            
            // Update command state when selection changes
            PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(SelectedObject))
                {
                    (RemoveObjectCommand as RelayCommand)?.RaiseCanExecuteChanged();
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
            SelectedObject = obj;
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
                    Objects.RemoveAt(index);
                    SelectedObject = Objects.FirstOrDefault();
                    StatusMessage = "Object removed";
                }
                else
                {
                    StatusMessage = "Failed to remove object";
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

            StatusMessage = "Rendering...";
            
            try
            {
                int result = RendererAPI.render_scene(_sceneHandle, OutputPath, RenderWidth, RenderHeight, Luminosity);
                if (result != 0)
                {
                    StatusMessage = $"Rendered to {OutputPath}";
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
                                    new FilePickerFileType("PPM Files") { Patterns = new[] { "*.ppm" } }
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
