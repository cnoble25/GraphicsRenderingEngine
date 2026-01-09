using System;
using GraphicsRendererUI;

namespace GraphicsRendererUI.Models
{
    public class SceneObject : ObservableObject, ISceneItem
    {
        private string _name = "";
        private ObjectType_API _objectType = ObjectType_API.OBJECT_TYPE_PYRAMID;
        private Vec3_API _position = new Vec3_API { x = 0, y = 0, z = 0 };
        private Vec3_API _scale = new Vec3_API { x = 0, y = 0, z = 0 };
        private Rotation_API _rotation = new Rotation_API { roll = 0, pitch = 0, yaw = 0 };
        private string _objFilePath = "";
        private bool _useDegrees = true; // Default to degrees
        private double _lightAbsorption = 1.0; // Default light absorption (1.0 = completely matte, 0.0 = perfect mirror)

        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }

        public ObjectType_API ObjectType
        {
            get => _objectType;
            set => SetProperty(ref _objectType, value);
        }

        public Vec3_API Position
        {
            get => _position;
            set => SetProperty(ref _position, value);
        }

        public Vec3_API Scale
        {
            get => _scale;
            set => SetProperty(ref _scale, value);
        }

        public Rotation_API Rotation
        {
            get => _rotation;
            set => SetProperty(ref _rotation, value);
        }

        public string ObjFilePath
        {
            get => _objFilePath;
            set => SetProperty(ref _objFilePath, value);
        }

        public double LightAbsorption
        {
            get => _lightAbsorption;
            set => SetProperty(ref _lightAbsorption, Math.Max(0.0, Math.Min(1.0, value))); // Clamp to [0, 1]
        }

        public double PositionX
        {
            get => _position.x;
            set
            {
                _position.x = value;
                OnPropertyChanged(nameof(PositionX));
                OnPropertyChanged(nameof(Position));
            }
        }

        public double PositionY
        {
            get => _position.y;
            set
            {
                _position.y = value;
                OnPropertyChanged(nameof(PositionY));
                OnPropertyChanged(nameof(Position));
            }
        }

        public double PositionZ
        {
            get => _position.z;
            set
            {
                _position.z = value;
                OnPropertyChanged(nameof(PositionZ));
                OnPropertyChanged(nameof(Position));
            }
        }

        public double ScaleX
        {
            get => _scale.x;
            set
            {
                _scale.x = value;
                OnPropertyChanged(nameof(ScaleX));
                OnPropertyChanged(nameof(Scale));
            }
        }

        public double ScaleY
        {
            get => _scale.y;
            set
            {
                _scale.y = value;
                OnPropertyChanged(nameof(ScaleY));
                OnPropertyChanged(nameof(Scale));
            }
        }

        public double ScaleZ
        {
            get => _scale.z;
            set
            {
                _scale.z = value;
                OnPropertyChanged(nameof(ScaleZ));
                OnPropertyChanged(nameof(Scale));
            }
        }

        public bool UseDegrees
        {
            get => _useDegrees;
            set
            {
                if (SetProperty(ref _useDegrees, value))
                {
                    // Convert rotation values when switching between degrees and radians
                    if (value)
                    {
                        // Converting from radians to degrees
                        _rotation.roll = _rotation.roll * 180.0 / Math.PI;
                        _rotation.pitch = _rotation.pitch * 180.0 / Math.PI;
                        _rotation.yaw = _rotation.yaw * 180.0 / Math.PI;
                    }
                    else
                    {
                        // Converting from degrees to radians
                        _rotation.roll = _rotation.roll * Math.PI / 180.0;
                        _rotation.pitch = _rotation.pitch * Math.PI / 180.0;
                        _rotation.yaw = _rotation.yaw * Math.PI / 180.0;
                    }
                    OnPropertyChanged(nameof(RotationRoll));
                    OnPropertyChanged(nameof(RotationPitch));
                    OnPropertyChanged(nameof(RotationYaw));
                    OnPropertyChanged(nameof(Rotation));
                }
            }
        }

        public double RotationRoll
        {
            get => _rotation.roll;
            set
            {
                _rotation.roll = value;
                OnPropertyChanged(nameof(RotationRoll));
                OnPropertyChanged(nameof(Rotation));
            }
        }

        public double RotationPitch
        {
            get => _rotation.pitch;
            set
            {
                _rotation.pitch = value;
                OnPropertyChanged(nameof(RotationPitch));
                OnPropertyChanged(nameof(Rotation));
            }
        }

        public double RotationYaw
        {
            get => _rotation.yaw;
            set
            {
                _rotation.yaw = value;
                OnPropertyChanged(nameof(RotationYaw));
                OnPropertyChanged(nameof(Rotation));
            }
        }

        public SceneObject_API ToAPI()
        {
            // Default scale to 1.0 if it's 0 (allows placeholder to show)
            var scale = Scale;
            if (scale.x == 0 && scale.y == 0 && scale.z == 0)
            {
                scale = new Vec3_API { x = 1.0, y = 1.0, z = 1.0 };
            }
            
            // Convert rotation to radians if currently in degrees (API expects radians)
            Rotation_API rotation = Rotation;
            if (_useDegrees)
            {
                rotation = new Rotation_API
                {
                    roll = Rotation.roll * Math.PI / 180.0,
                    pitch = Rotation.pitch * Math.PI / 180.0,
                    yaw = Rotation.yaw * Math.PI / 180.0
                };
            }
            
            var api = new SceneObject_API
            {
                type = ObjectType,
                transform = new Transform_API
                {
                    position = Position,
                    scale = scale,
                    rotation = rotation
                },
                obj_file_path = (ObjectType == ObjectType_API.OBJECT_TYPE_OBJ_FILE && !string.IsNullOrEmpty(ObjFilePath)) 
                    ? ObjFilePath 
                    : null,
                light_absorption = LightAbsorption
            };

            return api;
        }
    }
}
