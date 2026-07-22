#pragma once

#include <soloud.h>
#include <soloud_wav.h>
#include <soloud_wavstream.h>
#include <memory>
#include <unordered_map>
#include <string>
#include <glm/glm.hpp>

enum class SoundType {
    Music,
    SFX,
    UI
};

enum class SoundID {
    BackgroundMusic,
    MainMenuTheme,
    BossTheme,

    ButtonClick,
    ButtonHover,
    MenuClose,

    Collision,
    WaterSplash,
    Fire,
    Crunch,
    Step,
    Run,
    Jump,
    Land,
    Suffocate
};

class SoundManager {
public:
    static SoundManager& getInstance() {
        static SoundManager instance;
        return instance;
    }

    SoundManager(const SoundManager&) = delete;
    SoundManager& operator=(const SoundManager&) = delete;

    bool init();
    void shutdown();
    void update();

    void setMasterVolume(float volume);
    void setSFXVolume(float volume);
    void setMusicVolume(float volume);

    float getMasterVolume() const { return masterVolume; }
    float getSFXVolume() const { return sfxVolume; }
    float getMusicVolume() const { return musicVolume; }

    bool loadSound(SoundID id, const std::string& filepath, SoundType type = SoundType::SFX);
    bool loadMusic(SoundID id, const std::string& filepath);

    SoLoud::handle playSound(SoundID id, float volume = 1.0f, float pitch = 1.0f, bool adaptPitch = true, bool allowStacking = false, int maxStackAmount = 0);
    SoLoud::handle playSound3D(SoundID id, const glm::vec3& position,
                               float volume = 1.0f, float pitch = 1.0f);
    SoLoud::handle playMusic(SoundID id, bool loop = true, float volume = 1.0f);

    void stopSound(SoLoud::handle handle);
    void stopAllSounds();
    void stopMusic();
    void stopSoundByID(SoundID id);

    void pauseSound(SoLoud::handle handle);
    void resumeSound(SoLoud::handle handle);
    void pauseAll();
    void resumeAll();

    void set3dListenerPosition(float x, float y, float z);
    void set3dListenerPosition(const glm::vec3& position);
    void set3dListenerParameters(const glm::vec3& position, const glm::vec3& velocity,
                                 const glm::vec3& forward, const glm::vec3& up);
    void set3dListenerVelocity(const glm::vec3& velocity);
    void update3dAudio();

    bool isSoundLoaded(SoundID id) const;
    bool isSoundPlaying(SoLoud::handle handle);
    float getSoundVolume(SoLoud::handle handle);
    SoLoud::Soloud& getAudioEngine() { return soloud; }

    void setSoundVolume(SoLoud::handle handle, float volume);
    void setSoundPitch(SoLoud::handle handle, float pitch);
    void setSound3DParameters(SoLoud::handle handle, const glm::vec3& position,
                              float minDistance = 1.0f, float maxDistance = 100.0f);
    void setSound3DVelocity(SoLoud::handle handle, const glm::vec3& velocity);

    void setGlobalSoundVolume(float volume);
    void set3DAttenuation(SoLoud::AudioSource& source,
                          unsigned int model,
                          float attenuationFactor = 1.0f);

private:
    SoundManager();
    ~SoundManager();

    float calculateFinalVolume(SoundType type, float baseVolume) const;

    SoLoud::handle playSoundInternal(SoundID id,
                                     SoLoud::AudioSource* source,
                                     SoundType type,
                                     float volume,
                                     float pitch,
                                     const glm::vec3* position = nullptr);

    void refreshPlayingVolumes();

    SoLoud::Soloud soloud;

    struct SoundEntry {
        std::unique_ptr<SoLoud::AudioSource> source;
        SoundType type = SoundType::SFX;
        bool isLoaded = false;
        std::string filepath;
    };

    std::unordered_map<SoundID, SoundEntry> sounds;
    std::unordered_map<SoLoud::handle, SoundID> activeHandles;

    float masterVolume = 1.0f;
    float sfxVolume = 1.0f;
    float musicVolume = 0.8f;

    SoLoud::handle currentMusicHandle = 0;
    SoundID currentMusicID = SoundID::BackgroundMusic;

    glm::vec3 listenerPosition = glm::vec3(0.0f);
    glm::vec3 listenerVelocity = glm::vec3(0.0f);
    glm::vec3 listenerForward = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 listenerUp = glm::vec3(0.0f, 1.0f, 0.0f);

    bool initialized = false;
};

#define g_SoundManager SoundManager::getInstance()